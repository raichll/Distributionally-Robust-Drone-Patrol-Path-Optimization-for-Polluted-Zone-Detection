import vrplib
import numpy as np
import torch
import yaml
import json
import time
import os
from torch.optim import Adam as Optimizer
from CVRPModel_DRO import CVRPModel_DRO
from CVRPEnv_DRO import CVRPEnv_DRO
from utils_dro import rollout_dro, check_feasible
import warnings

# 忽略 RuntimeWarning
warnings.simplefilter("ignore", category=RuntimeWarning)

class VRPLib_Tester_DRO:

    def __init__(self, config):
        self.config = config
        model_params = config['model_params']
        load_checkpoint = config['load_checkpoint']

        # DRO parameters
        self.dro_params = config.get('dro_params', {
            'epsilon': 0.1,      # Wasserstein ball radius
            'M': 1.0,         # Big-M penalty for unvisited nodes
            'N': 50,             # Number of historical samples
            'lambda_reg': 0.01,  # Regularization for dual variables
            'uncertainty_std': 0.1,  # Standard deviation for demand uncertainty
            'sample_strategy': 'uniform'  # 'gaussian', 'uniform', or 'historical'
        })

        # cuda
        USE_CUDA = config.get('use_cuda', False)
        if USE_CUDA:
            cuda_device_num = config['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            self.device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            self.device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')
        
        # load trained model or initialize new one
        self.model = CVRPModel_DRO(**model_params)
        


        if model_params.get('ensemble', False):
            self.model.decoder.add_local_policy(self.device)
               
        checkpoint = torch.load(load_checkpoint, map_location=self.device)
        
        # 获取模型的 state_dict
        model_dict = self.model.state_dict()
        checkpoint_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        

        # 比较模型和检查点的参数
        for key in checkpoint_dict:
            if key in model_dict:
                if checkpoint_dict[key].shape != model_dict[key].shape:
                    print(f"形状不匹配: {key}")
                    print(f"检查点的形状: {checkpoint_dict[key].shape}, 当前模型的形状: {model_dict[key].shape}")
            else:
                print(f"检查点中有模型中没有的参数: {key}")
       

           


                       
        self.vrplib_path = 'VRPLib/Vrp-Set-X/' if config['vrplib_set'] == 'X' else "VRPLib/Vrp-Set-XXL/"
        self.repeat_times = config.get('repeat_times', 1)
        self.aug_factor = config['params']['aug_factor']
        self.vrplib_results = None
        
    def generate_historical_samples(self, base_demand, instance_name=None):
        """Generate historical demand samples for DRO"""
        N = self.dro_params['N']
        batch_size, problem_size = base_demand.shape
        
        if self.dro_params['sample_strategy'] == 'gaussian':
            # Generate Gaussian noise around base demand
            std = self.dro_params['uncertainty_std']
            samples = []
            for _ in range(N):
                noise = torch.randn_like(base_demand) * std
                sample = torch.clamp(base_demand + noise, min=0.01, max=1.0)
                samples.append(sample)
                
        elif self.dro_params['sample_strategy'] == 'uniform':
            # Generate uniform samples around base demand
            deviation = self.dro_params['uncertainty_std']
            samples = []
            for _ in range(N):
                noise = (torch.rand_like(base_demand) - 0.5) * 2 * deviation
                sample = torch.clamp(base_demand + noise, min=0.01, max=1.0)
                samples.append(sample)
                
        elif self.dro_params['sample_strategy'] == 'historical':
            # Use predefined historical samples (you would load these from data)
            # For now, we'll simulate this with a mix of patterns
            samples = []
            for i in range(N):
                # Create different demand patterns
                if i % 4 == 0:  # High demand scenario
                    sample = torch.clamp(base_demand * (1.0 + 0.3 * torch.rand_like(base_demand)), min=0.01, max=1.0)
                elif i % 4 == 1:  # Low demand scenario
                    sample = torch.clamp(base_demand * (0.7 + 0.3 * torch.rand_like(base_demand)), min=0.01, max=1.0)
                elif i % 4 == 2:  # Clustered high demand
                    sample = base_demand.clone()
                    # Make some nodes have higher demand
                    high_demand_nodes = torch.randperm(problem_size)[:problem_size//3]
                    sample[:, high_demand_nodes] *= 1.5
                    sample = torch.clamp(sample, min=0.01, max=1.0)
                else:  # Normal variation
                    noise = torch.randn_like(base_demand) * 0.1
                    sample = torch.clamp(base_demand + noise, min=0.01, max=1.0)
                samples.append(sample)
        
        historical_samples = torch.stack(samples, dim=1)  # shape: (batch, N, problem)
        
        return historical_samples
        
    def test_on_vrplib(self):
        files = os.listdir(self.vrplib_path)
        vrplib_results = []
        total_time = 0.
        lis_time = []
        lis_gap = []
        
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
                self.test_on_one_ins(name=name, result_dict=result_dict, 
                                   instance=instance_file, solution=solution_file)
                total_time += time.time() - start_time
                lis_time.append(time.time() - start_time)
                
                new_instance_dict = {}
                new_instance_dict['instance'] = name
                new_instance_dict['optimal'] = optimal
                new_instance_dict['record'] = [result_dict]
                vrplib_results.append(new_instance_dict)
                lis_gap.append(result_dict['gap'])
                
                print(f"Instance Name {name}: gap {result_dict['gap']:.4f}, "
                      f"dro_cost {result_dict.get('dro_cost', 'N/A'):.4f}, "
                      f"routing_cost {result_dict.get('routing_cost', 'N/A'):.4f}")
                print('=========================================')
                
                if 'XXL' in self.vrplib_path:
                    print("cost: {}".format(result_dict['best_cost']))
                    
        # Results analysis
        if 'XXL' in self.vrplib_path:
            avg_gap = []
            for result in vrplib_results:
                avg_gap.append(result['record'][-1]['gap'])
            
            print("{:.2f}%".format(100 * np.array(avg_gap).mean()))
            print("Average time: {:.2f}s".format(total_time / len(vrplib_results)))
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
            
            print("DRO-CVRP Results:")
            print("Average gap on subset of <200: {:.2f}%".format(100 * np.array(avg_gap_small).mean()))
            print("Average gap on subset of 200-500: {:.2f}%".format(100 * np.array(avg_gap_medium).mean()))
            print("Average gap on subset of 500-1000: {:.2f}%".format(100 * np.array(avg_gap_large).mean()))
            print("Average gap total: {:.2f}%".format(100 * (np.array(total).mean())))
            print("Average time: {:.2f}s".format(total_time / number))
            print("Total time: {:.2f}s".format(total_time))
            
            # Additional DRO-specific statistics
            dro_costs = [result['record'][-1].get('dro_cost', 0) for result in vrplib_results]
            routing_costs = [result['record'][-1].get('routing_cost', 0) for result in vrplib_results]
            print(f"Average DRO penalty: {np.mean(dro_costs):.2f}")
            print(f"Average routing cost: {np.mean(routing_costs):.2f}")
            
            vrplib_results.append({
                "<200": 100 * np.array(avg_gap_small).mean(), 
                "200-1000": 100 * np.array(avg_gap_large).mean(),
                "total": 100 * (np.array(total).mean()),
                "dro_params": self.dro_params
            })
            
            # 确保目录存在
            os.makedirs('test_results', exist_ok=True)
            with open('test_results/' + self.config['name'] + '_dro_vrplib.json', 'w') as f:
                json.dump(vrplib_results, f, indent=2)

    def test_on_one_ins(self, name, result_dict, instance, solution):
        instance = vrplib.read_instance(instance)
        solution = vrplib.read_solution(solution)
        
        optimal = solution['cost']
        problem_size = instance['node_coord'].shape[0] - 1
        multiple_width = min(problem_size, 1000)

        # Generate historical samples for this instance
        base_demand = torch.FloatTensor(instance['demand']).unsqueeze(0) / instance['capacity']
        historical_samples = self.generate_historical_samples(base_demand[:, 1:], name)  # Exclude depot
        
        # Initialize DRO-CVRP environment
        env = CVRPEnv_DRO(multiple_width, self.device, self.dro_params)
        env.load_vrplib_problem(instance, aug_factor=self.aug_factor, 
                               historical_samples=historical_samples)
        
        empirical_dist = torch.randint(low=0, high=2, size=(100,))  # 0表示未访问，1表示已访问
        
        env.add_empirical_distribution(empirical_dist)
        
        reset_state, reward, done = env.reset()
        self.model.eval()
        self.model.requires_grad_(False)
        self.model.pre_forward(reset_state)
        
        with torch.no_grad():
            policy_solutions, policy_prob, rewards = rollout_dro(self.model, env, 'greedy')
        
        # Process results
        aug_reward = rewards.reshape(self.aug_factor, 1, env.multi_width)
        
        # shape: (augmentation, batch, multi)
        max_pomo_reward, pomo_indices = aug_reward.max(dim=2)  # get best results from pomo
        
        # shape: (augmentation, batch)
        max_aug_pomo_reward, aug_indices = max_pomo_reward.max(dim=0)  # get best results from augmentation
        
        # shape: (batch,)
        aug_cost = -max_aug_pomo_reward.float()  # negative sign to make positive value
        
        best_cost = aug_cost
        
        # Separate routing cost and DRO penalty for analysis
        best_augmentation_idx = aug_indices.item()
        best_pomo_idx = pomo_indices[best_augmentation_idx, 0].item()
        
        # Get routing cost without DRO penalty
        routing_cost = env.compute_unscaled_routing_cost_only()
        
        best_routing_cost = -routing_cost.reshape(self.aug_factor, 1, env.multi_width).max(dim=2)[0].max(dim=0)[0].float()
        
        # DRO penalty is the difference
        dro_penalty = best_cost - best_routing_cost
        
        if result_dict is not None:
            result_dict['best_cost'] = best_cost.cpu().numpy().tolist()[0]
            result_dict['routing_cost'] = best_routing_cost.cpu().numpy().tolist()[0]
            result_dict['dro_cost'] = dro_penalty.cpu().numpy().tolist()[0]
            result_dict['scale'] = problem_size
            result_dict['gap'] = (result_dict['best_cost'] - optimal) / optimal
            result_dict['dro_params'] = self.dro_params
            
        # Get the best solution path
        final_best_solution_path = policy_solutions[best_augmentation_idx, best_pomo_idx, :]

        print(f"DRO analysis for {name}:")
        print(f"  Total cost: {best_cost.item():.4f}")
        print(f"  Routing cost: {best_routing_cost.item():.4f}")
        print(f"  DRO penalty: {dro_penalty.item():.4f}")
        print(f"  Best augmentation idx: {best_augmentation_idx}")
        print(f"  Best POMO idx: {best_pomo_idx}")
        print(f"  Solution path length: {final_best_solution_path}")
        
        return result_dict


if __name__ == "__main__":
    
    # 定义目标目录路径
    target_directory = r"elg-master\cvrp_dro"

    # 使用 os.chdir() 切换到目标目录
    os.chdir(target_directory)
    
    # Load configuration
    with open('config_dro.yml', 'r', encoding='utf-8') as config_file:
        config = yaml.load(config_file.read(), Loader=yaml.FullLoader)
    
    tester = VRPLib_Tester_DRO(config=config)
    tester.test_on_vrplib()