import vrplib
import numpy as np
import torch
import yaml
import json
import time
import os
from torch.optim import Adam as Optimizer
from KLH import LKHOptimizer
from CVRPModel import CVRPModel, CVRPModel_local
from CVRPEnv import CVRPEnv
from utils import rollout, check_feasible
import warnings

# 忽略 RuntimeWarning
warnings.simplefilter("ignore", category=RuntimeWarning)

class VRPLib_Tester:

    def __init__(self, config):
        self.config = config
        model_params = config['model_params']
        load_checkpoint = config['load_checkpoint']

        # cuda
        #USE_CUDA = config['use_cuda']
        USE_CUDA=False
        if USE_CUDA:
            cuda_device_num = config['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            self.device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            self.device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')
        
        # load trained model
        self.model = CVRPModel(**model_params)
        if model_params['ensemble']:
            self.model.decoder.add_local_policy(self.device)

        checkpoint = torch.load(load_checkpoint, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.vrplib_path = 'VRPLib/Vrp-Set-X/' if config['vrplib_set'] == 'X' else "VRPLib/Vrp-Set-XXL/"
        self.repeat_times = 1
        self.aug_factor = config['params']['aug_factor']
        self.vrplib_results = None
        
    def test_on_vrplib(self):
        files = os.listdir(self.vrplib_path)
        vrplib_results = []
        total_time = 0.
        lis_time= []
        lis_gap= []
        for t in range(self.repeat_times):
            for name in files:
                if '.sol' in name:
                    continue
                name = name[:-4]
                instance_file = self.vrplib_path + '/' + name + '.vrp'
                solution_file = self.vrplib_path + '/' + name + '.sol'
                
                solution = vrplib.read_solution(solution_file)
                optimal = solution['cost']
                
                result_dict = {}
                result_dict['run_idx'] = t
                start_time = time.time()
                self.test_on_one_ins(name=name, result_dict=result_dict, instance=instance_file, solution=solution_file)
                total_time += time.time() - start_time
                lis_time.append(time.time() - start_time)
                new_instance_dict = {}
                new_instance_dict['instance'] = name
                new_instance_dict['optimal'] = optimal
                new_instance_dict['record'] = [result_dict]
                vrplib_results.append(new_instance_dict)
                lis_gap.append(result_dict['gap'])
                print("Instance Name {}: gap {:.4f}".format(name, result_dict['gap']))
                print('=========================================')
                if 'XXL' in self.vrplib_path:
                    print("cost: {}".format(result_dict['best_cost']))
        if 'XXL' in self.vrplib_path:
            avg_gap = []
            for result in vrplib_results:
                avg_gap.append(result['record'][-1]['gap'])
            
            print("{:.2f}%".format(100 * np.array(avg_gap).mean()))
            print("Average time: {:.2f}s".format(total_time / 4))
        else:
            avg_gap_small = []
            avg_gap_medium = []
            avg_gap_large = []
            total = []
            number = 0
            for result in vrplib_results:
                scale = int(result['record'][-1]['scale'])
                if scale <= 200:
                    avg_gap_small.append(result['record'][-1]['gap'])
                elif scale <= 500:
                    avg_gap_medium.append(result['record'][-1]['gap'])
                else:
                    avg_gap_large.append(result['record'][-1]['gap'])
                total.append(result['record'][-1]['gap'])
                number += 1
            
            print("Average gap on subset of <200: {:.2f}%".format(100 * np.array(avg_gap_small).mean()))
            print("Average gap on subset of 200-500: {:.2f}%".format(100 * np.array(avg_gap_medium).mean()))
            print("Average gap on subset of 500-1000: {:.2f}%".format(100 * np.array(avg_gap_large).mean()))
            print("Average gap total: {:.2f}%".format(100 *(np.array(total).mean())))
            print("Average time: {:.2f}s".format(total_time / number))
            print("total time: {:.2f}s".format(total_time))
            print("gap: ", lis_gap)
            print("time: ", lis_time)
            vrplib_results.append({"<200": 100 * np.array(avg_gap_small).mean(), 
            "200-1000": 100 * np.array(avg_gap_large).mean(),
            "total": 100 *(np.array(total).mean())})
            # 确保目录存在
            os.makedirs('test_results', exist_ok=True)
            with open('test_results/' + self.config['name'] + '_' + 'vrplib.json', 'w') as f:
                json.dump(vrplib_results, f)


    def test_on_one_ins(self, name, result_dict, instance, solution):
        instance = vrplib.read_instance(instance)
        solution = vrplib.read_solution(solution)
        
        optimal = solution['cost']
        problem_size = instance['node_coord'].shape[0] - 1
        #multiple_width = min(problem_size, 1000)
        multiple_width=1
        # multiple_width = problem_size

        # Initialize CVRP state
        env = CVRPEnv(multiple_width, self.device)
        env.load_vrplib_problem(instance, aug_factor=self.aug_factor)
        
        reset_state, reward, done = env.reset()
        self.model.eval()
        self.model.requires_grad_(False)
        self.model.pre_forward(reset_state)
        
        with torch.no_grad():
            policy_solutions, policy_prob, rewards = rollout(self.model, env, 'greedy')
        print(policy_solutions.shape)
        
        # Return
        aug_reward = rewards.reshape(self.aug_factor, 1, env.multi_width)
        
        # shape: (augmentation, batch, multi)
        max_pomo_reward, _ = aug_reward.max(dim=2)  # get best results from pomo
        
        # shape: (augmentation, batch)
        max_aug_pomo_reward, _ = max_pomo_reward.max(dim=0)  # get best results from augmentation
        
        # shape: (batch,)
        aug_cost = -max_aug_pomo_reward.float()  # negative sign to make positive value
        
        best_cost = aug_cost
        
        if result_dict is not None:
            result_dict['best_cost'] = best_cost.cpu().numpy().tolist()[0]
            result_dict['scale'] = problem_size
            result_dict['gap'] = (result_dict['best_cost'] - optimal) / optimal
            #print(best_cost)
        
                
        # 示例路径长度# 1. 重新塑形 rewards，使其形状为 (augmentation, batch=1, multi_width)
        # 这里的 batch 始终是 1，因为 test_on_one_ins 每次处理一个 VRPLib 问题实例
        aug_reward = rewards.reshape(self.aug_factor, 1, env.multi_width)
        # 形状: (augmentation, 1, multi_width) -> 例如 (8, 1, 100)

        # 2. 从 POMO 维度中获取最佳结果（最高奖励值）及其索引
        # max(dim=2) 在 multi_width 维度上取最大值
        max_pomo_reward, pomo_indices = aug_reward.max(dim=2)
        
        # max_pomo_reward 形状: (augmentation, 1) -> 例如 (8, 1)
        # pomo_indices 形状: (augmentation, 1) -> 例如 (8, 1)，包含每个 augmentation 下最佳 POMO 的索引

        # 3. 从数据增强（Augmentation）维度中获取最佳结果（最高奖励值）及其索引
        # max(dim=0) 在 augmentation 维度上取最大值
        max_aug_pomo_reward, aug_indices = max_pomo_reward.max(dim=0)
        # max_aug_pomo_reward 形状: (1,) -> 例如 (1,)，这就是最终的最高奖励值
        # aug_indices 形状: (1,) -> 例如 (1,)，包含最佳 augmentation 的索引

        # 4. 获取对应的具体索引值
        best_augmentation_idx = aug_indices.item() # 获取最佳 augmentation 的索引 (0-7)
        # 从 pomo_indices 中获取对应最佳 augmentation 的 POMO 索引
        best_pomo_idx = pomo_indices[best_augmentation_idx, 0].item() # 获取最佳 POMO 的索引 (0-99)

        # 5. 使用这两个索引从 policy_solutions 中提取出最终的最优解路径
        final_best_solution_path = policy_solutions[best_augmentation_idx, best_pomo_idx, :]

        # final_best_solution_path 的形状将是 (tour_length,) -> 例如 (131,)
        # 这就是对应 best_cost 的那条具体的路径（节点序列）

        # 最终的 best_cost（正值）
        final_best_cost = -max_aug_pomo_reward.float().item()

        print(f"最终计算出的 best_cost (路径长度): {final_best_cost:.4f}")
        print(f"对应这个 best_cost 的数据增强索引 (augmentation index): {best_augmentation_idx}")
        print(f"对应这个 best_cost 的 POMO 索引 (multi_width index): {best_pomo_idx}")
        print(f"最终最优解的路径: {final_best_solution_path}")
        print(f"最终最优解的路径完整形状: {final_best_solution_path.shape}")
        




if __name__ == "__main__":
    
     # 定义目标目录路径
    target_directory = r"elg-master\cvrp"

    # 使用 os.chdir() 切换到目标目录
    os.chdir(target_directory)
    
    with open('config.yml', 'r', encoding='utf-8') as config_file:
        config = yaml.load(config_file.read(), Loader=yaml.FullLoader)
    tester = VRPLib_Tester(config=config)
    tester.test_on_vrplib()