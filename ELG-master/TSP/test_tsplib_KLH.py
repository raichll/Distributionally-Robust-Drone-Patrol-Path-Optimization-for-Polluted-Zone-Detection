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
from KLH import LKHOptimizer


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
        
        # 找到最佳解
        aug_reward = rewards.reshape(self.aug_factor, 1, pomo_size)
        max_pomo_reward, pomo_indices = aug_reward.max(dim=2)
        max_aug_pomo_reward, aug_indices = max_pomo_reward.max(dim=0)
        
        best_augmentation_idx = aug_indices.item()
        best_pomo_idx = pomo_indices[best_augmentation_idx, 0].item()
        
        # 提取最佳解路径 - policy_solutions shape: [aug_factor, pomo_size, problem_size]
        stage1_solution = policy_solutions[best_augmentation_idx][best_pomo_idx]
        stage1_cost = -max_aug_pomo_reward.float().item()
        
        print(f"第一阶段RL成本: {stage1_cost:.4f}")
        
        # 第二阶段：LKH优化
        try:
            # 创建LKH优化器
            coordinates = instance[0].tolist()
            lkh_optimizer = LKHOptimizer(distance_matrix=np.array([]), demands=[], vehicle_capacity=float('inf'))
            distance_matrix = lkh_optimizer.calculate_distance_matrix(coordinates)
            
            lkh_optimizer.distance_matrix = distance_matrix
            lkh_optimizer.demands = [0] * problem_size
            lkh_optimizer.vehicle_capacity = float('inf')
            lkh_optimizer.depot = 0
            
            # 转换解格式
            initial_solution = [stage1_solution.cpu().numpy().tolist()]
            
            # LKH优化
            optimized_routes, final_cost = lkh_optimizer.lkh_optimize(
                initial_solution, 
                op=optimal,
                max_iterations=50
            )
            
            print(f"第二阶段LKH成本: {final_cost:.4f}")
            
        except Exception as e:
            print(f"LKH优化失败: {e}")
            final_cost = stage1_cost
        
        if result_dict is not None:
            result_dict['best_cost'] = final_cost
            result_dict['scale'] = problem_size
            result_dict['gap'] = (final_cost - optimal) / optimal


if __name__ == "__main__":
    # 定义目标目录路径
    target_directory = r"elg-master\tsp"

    # 使用 os.chdir() 切换到目标目录
    os.chdir(target_directory)
    with open('config.yml', 'r', encoding='utf-8') as config_file:
        config = yaml.load(config_file.read(), Loader=yaml.FullLoader)
    tester = TSPLib_Tester(config=config)
    tester.test_on_tsplib()