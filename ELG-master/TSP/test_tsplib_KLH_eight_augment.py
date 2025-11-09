import os
import yaml
import time
import pickle
import json
import torch
import numpy as np
from torch.optim import Adam as Optimizer

from generate_data import generate_tsp_data, TSPDataset
from TSPModel import TSPModel, Att_Local_policy
from TSPEnv import TSPEnv
from utils import rollout, batched_two_opt_torch, check_feasible
from KLH import LKHOptimizer, stage2_tsp_lkh_optimization


class TSPLib_Tester:

    def __init__(self, config):
        self.config = config
        model_params = config['model_params']
        load_checkpoint = config['load_checkpoint']

        self.device = torch.device('cpu')
        torch.set_default_tensor_type('torch.FloatTensor')
        
        # load trained model
        if config['training'] == 'joint':
            self.model = TSPModel(**model_params)
            if model_params['ensemble']:
                self.model.decoder.add_local_policy(self.device)
        elif config['training'] == 'only_local_att':
            self.model = Att_Local_policy(**model_params)

        checkpoint = torch.load(load_checkpoint, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.tsplib_path = 'TSPLib'
        self.repeat_times = 1
        self.aug_factor = config['params']['aug_factor']
        
    def test_on_tsplib(self):
        files = os.listdir(self.tsplib_path)
        tsplib_results = []
        total_time = 0.
        lis_time = []
        lis_gap = []
        
        for t in range(self.repeat_times):
            for name in files:
                if '.sol' in name:
                    continue
                name = name[:-4]
                instance_file = self.tsplib_path + '/' + name + '.pkl'
                
                print(instance_file)
                with open(instance_file, 'rb') as f:
                    instance = pickle.load(f)  
                    optimal = instance[1]

                result_dict = {}
                result_dict['run_idx'] = t
                start_time = time.time()
                self.test_on_one_ins(name=name, result_dict=result_dict, instance=instance)
                elapsed_time = time.time() - start_time
                total_time += elapsed_time
                lis_time.append(elapsed_time)

                new_instance_dict = {}
                new_instance_dict['instance'] = name
                new_instance_dict['optimal'] = optimal
                new_instance_dict['record'] = [result_dict]
                tsplib_results.append(new_instance_dict)
                lis_gap.append(result_dict['gap'])
                
                print("Instance Name {}: gap {:.4f}%".format(name, result_dict['gap'] * 100))
                print('=========================================')

        # 统计结果
        avg_gap_small = []
        avg_gap_medium = []
        avg_gap_large = []
        total = []
        number = 0
        
        for result in tsplib_results:
            scale = int(result['record'][-1]['scale'])
            gap = result['record'][-1]['gap']
            if scale <= 200:
                avg_gap_small.append(gap)
            elif scale <= 500:
                avg_gap_medium.append(gap)
            else:
                avg_gap_large.append(gap)
            total.append(gap)
            number += 1
        
        print("Average gap on subset of <=200: {:.2f}%".format(100 * np.array(avg_gap_small).mean()))
        print("Average gap on subset of 200-500: {:.2f}%".format(100 * np.array(avg_gap_medium).mean()))
        print("Average gap on subset of 500-1002: {:.2f}%".format(100 * np.array(avg_gap_large).mean()))
        print("Average gap total: {:.2f}%".format(100 * np.array(total).mean()))
        print("Average time: {:.2f}s".format(total_time / number))
        print("Total time: {:.2f}s".format(total_time))

    def test_on_one_ins(self, name, result_dict, instance):
        unscaled_points = torch.tensor(instance[0], dtype=torch.float)[None, :, :]
        points = (instance[0] - np.min(instance[0])) / (np.max(instance[0]) - np.min(instance[0]))
        test_batch = torch.tensor(points, dtype=torch.float)[None, :, :]
        optimal = instance[1]

        problem_size = test_batch.shape[1]
        pomo_size = problem_size
        batch_size = test_batch.shape[0]

        # initialize env
        env = TSPEnv(pomo_size, self.device)
        env.load_tsplib_problem(test_batch, unscaled_points, self.aug_factor)
        reset_state, reward, done = env.reset()

        self.model.eval()
        self.model.requires_grad_(False)
        self.model.pre_forward(reset_state)

        # 第一阶段：强化学习求解
        with torch.no_grad():
            policy_solutions, policy_prob, rewards = rollout(self.model, env, 'greedy')
        
        # ========== 修改部分：使用8次增强进行第二阶段优化 ==========
        print(f"开始二阶段优化，使用所有 {self.aug_factor} 个 augmentation 的最佳解决方案")
        
        # 1. 重新塑形 rewards，使其形状为 (augmentation, batch=1, pomo_size)
        aug_reward = rewards.reshape(self.aug_factor, 1, pomo_size)
        
        # 2. 从每个 augmentation 的 POMO 维度中获取最佳结果
        max_pomo_reward, pomo_indices = aug_reward.max(dim=2)
        # max_pomo_reward 形状: (augmentation, 1)
        # pomo_indices 形状: (augmentation, 1)
        
        # 3. 获取所有 augmentation 中的最佳结果（用于比较）
        max_aug_pomo_reward, aug_indices = max_pomo_reward.max(dim=0)
        stage1_best_cost = -max_aug_pomo_reward.float().item()
        
        print(f"第一阶段最佳成本: {stage1_best_cost:.4f}")
        
        # 4. 准备存储第二阶段优化结果的列表
        stage2_solutions = []
        stage2_costs = []
        
        # 5. 对每个 augmentation 的最佳解决方案进行第二阶段 LKH 优化
        for aug_idx in range(self.aug_factor):
            # 获取当前 augmentation 的最佳 POMO 索引
            best_pomo_idx = pomo_indices[aug_idx, 0].item()
            
            # 获取当前 augmentation 的最佳解决方案路径
            stage1_solution = policy_solutions[aug_idx, best_pomo_idx, :]
            
            # 获取当前 augmentation 的最佳成本
            stage1_cost = -max_pomo_reward[aug_idx, 0].float().item()
            
            print(f"Aug {aug_idx}: Stage1 cost = {stage1_cost:.4f}, POMO index = {best_pomo_idx}")
            
            try:
                # 第二阶段：TSP-LKH优化
                # 使用新的TSP专用优化函数，与TSPEnv距离计算方法完全一致
                coordinates = instance[0].tolist()
                
                # 使用TSPLib方法（四舍五入）进行距离计算，与TSPEnv的compute_unscaled_distance一致
                optimized_solution, stage2_cost = stage2_tsp_lkh_optimization(
                    coordinates=coordinates,
                    stage1_solution=stage1_solution,
                    optimal_cost=optimal,
                    use_tsplib_method=True  # 使用TSPLib的距离计算方法
                )
                
                stage2_solutions.append(optimized_solution)
                stage2_costs.append(stage2_cost)
                
                print(f"Aug {aug_idx}: Stage2 cost = {stage2_cost:.4f}")
                
            except Exception as e:
                print(f"Aug {aug_idx}: Stage2 optimization failed: {e}")
                # 如果第二阶段优化失败，使用第一阶段的结果和成本
                stage2_solutions.append(stage1_solution.cpu().numpy().tolist())
                stage2_costs.append(stage1_cost)
        
        # 6. 从所有第二阶段优化结果中选择最佳的
        best_stage2_idx = np.argmin(stage2_costs)
        final_best_cost = stage2_costs[best_stage2_idx]
        final_best_solution = stage2_solutions[best_stage2_idx]
        
        print(f"\n=== 二阶段优化结果总结 ===")
        print(f"最佳第二阶段结果来自 augmentation {best_stage2_idx}")
        print(f"最终优化成本: {final_best_cost:.4f}")
        print(f"相比第一阶段最佳成本 {stage1_best_cost:.4f} 的改进: {stage1_best_cost - final_best_cost:.4f}")
        print(f"最终优化解长度: {len(final_best_solution)}")
        
        # 7. 验证最终解的有效性（可选）
        if len(final_best_solution) == problem_size:
            print("解的长度验证通过")
        else:
            print(f"警告：解的长度 {len(final_best_solution)} 不等于问题规模 {problem_size}")
        
        # 8. 更新结果字典
        if result_dict is not None:
            result_dict['best_cost'] = final_best_cost
            result_dict['scale'] = problem_size
            result_dict['gap'] = (final_best_cost - optimal) / optimal
            result_dict['stage1_best_cost'] = stage1_best_cost
            result_dict['stage2_improvement'] = stage1_best_cost - final_best_cost
            result_dict['best_aug_idx'] = best_stage2_idx
            
            # 添加详细的每个augmentation结果
            result_dict['aug_stage1_costs'] = [-max_pomo_reward[i, 0].float().item() for i in range(self.aug_factor)]
            result_dict['aug_stage2_costs'] = stage2_costs.copy()
            
            print(f"第二阶段优化后的 gap: {result_dict['gap']:.4f}")


if __name__ == "__main__":
    # 定义目标目录路径
    target_directory = r"elg-master\tsp"

    # 使用 os.chdir() 切换到目标目录
    os.chdir(target_directory)
    with open('config.yml', 'r', encoding='utf-8') as config_file:
        config = yaml.load(config_file.read(), Loader=yaml.FullLoader)
    tester = TSPLib_Tester(config=config)
    tester.test_on_tsplib()