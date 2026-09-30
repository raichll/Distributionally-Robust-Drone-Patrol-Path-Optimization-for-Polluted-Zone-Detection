import vrplib
import numpy as np
import torch
import yaml
import json
import time
import os
import pandas as pd
from torch.optim import Adam as Optimizer
from KLH import LKHOptimizer
from CVRPModel import CVRPModel, CVRPModel_local
from CVRPEnv import CVRPEnv
from utils import rollout, check_feasible
from typing import List, Tuple, Dict
from KLH import convert_solution_to_sequence, stage2_lkh_optimization, parse_stage1_solution  
import warnings

# 忽略 RuntimeWarning
warnings.simplefilter("ignore", category=RuntimeWarning)


class IterationRecorder:
    """记录每次迭代的结果"""
    def __init__(self, target_times):
        self.target_times = sorted(target_times)
        self.start_time = None
        self.iteration_records = []  # [{iteration, time, cost, gap}]
        self.current_best_cost = float('inf')
        self.current_best_solution = None
        self.optimal = None
        
    def start(self, optimal):
        self.start_time = time.time()
        self.current_best_cost = float('inf')
        self.optimal = optimal
        self.iteration_records = []
        
    def record_iteration(self, iteration, cost, solution=None, phase='KLH'):
        """记录每次迭代的结果"""
        if cost < self.current_best_cost:
            self.current_best_cost = cost
            self.current_best_solution = solution
        
        elapsed = time.time() - self.start_time if self.start_time else 0
        gap = ((self.current_best_cost - self.optimal) / self.optimal * 100) if self.optimal else 0
        
        record = {
            'iteration': iteration,
            'phase': phase,
            'time': round(elapsed, 3),
            'cost': round(self.current_best_cost, 2),
            'gap': round(gap, 2)
        }
        self.iteration_records.append(record)
        
        print(f"  [{phase}] 迭代{iteration} | 时间:{elapsed:.2f}s | cost:{self.current_best_cost:.2f} | gap:{gap:.2f}%")
    
    def get_results_at_timepoints(self):
        """获取各时间点的最佳结果"""
        results = {}
        
        for target_time in self.target_times:
            # 找到小于等于target_time的最后一条记录
            best_record = None
            for record in self.iteration_records:
                if record['time'] <= target_time:
                    best_record = record
                else:
                    break  # 因为记录是按时间顺序的
            
            if best_record:
                results[target_time] = {
                    'cost': best_record['cost'],
                    'gap': best_record['gap'],
                    'actual_time': best_record['time'],
                    'iteration': best_record['iteration']
                }
        
        return results
    
    def get_all_iterations(self):
        """获取所有迭代记录"""
        return self.iteration_records


class VRPLib_Tester_WithIterations:

    def __init__(self, config, time_points=None):
        self.config = config
        model_params = config['model_params']
        load_checkpoint = config['load_checkpoint']
        
        # 时间点设置
        if time_points is None:
            self.time_points = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
        else:
            self.time_points = time_points
        self.max_time = max(self.time_points)

        # cuda
        USE_CUDA = False
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
        self.aug_factor = config['params']['aug_factor']
        
        # 存储所有实例的结果
        self.all_results = {}  # {instance_name: {timepoint_results, iteration_records}}
        
    def test_on_vrplib(self):
        """测试所有VRPLib实例，记录各时间点和各迭代结果"""
        files = os.listdir(self.vrplib_path)
        
        print(f"\n{'='*60}")
        print(f"POMO+KLH 多时间点测试（记录每次迭代）")
        print(f"{'='*60}")
        print(f"时间点: {self.time_points}")
        print(f"最大时间: {self.max_time}秒")
        print(f"{'='*60}\n")
        
        for name in files:
            if '.sol' in name:
                continue
            name = name[:-4]
            instance_file = self.vrplib_path + '/' + name + '.vrp'
            solution_file = self.vrplib_path + '/' + name + '.sol'
            
            solution = vrplib.read_solution(solution_file)
            optimal = solution['cost']
            
            print(f"\n[实例] {name} (最优解: {optimal})")
            
            # 运行单个实例（记录所有迭代）
            instance_results = self.test_on_one_ins_with_iterations(
                name=name,
                instance=instance_file,
                solution=solution_file,
                optimal=optimal
            )
            
            self.all_results[name] = instance_results
            
            # 打印该实例在各时间点的结果
            print(f"  时间点结果:")
            for tp in self.time_points:
                if tp in instance_results['timepoint_results']:
                    res = instance_results['timepoint_results'][tp]
                    print(f"    {tp}s: gap={res['gap']:.2f}% (迭代{res['iteration']})")
            
            # 保存该实例的详细结果
            self.save_instance_results(name, instance_results)
            print('=========================================')
        
        # 生成汇总报告
        self.generate_summary()
        
        # 导出汇总Excel
        self.export_summary_to_excel()
    
    def test_on_one_ins_with_iterations(self, name, instance, solution, optimal):
        """测试单个实例，记录每次迭代的解"""
        instance = vrplib.read_instance(instance)
        problem_size = instance['node_coord'].shape[0] - 1
        multiple_width = 1
        
        # 初始化迭代记录器
        recorder = IterationRecorder(self.time_points)
        recorder.start(optimal)
        
        # =========================
        # 第一阶段：POMO
        # =========================
        print("  [POMO阶段] 开始...")
        pomo_start = time.time()
        
        # Initialize CVRP state
        env = CVRPEnv(multiple_width, self.device)
        env.load_vrplib_problem(instance, aug_factor=self.aug_factor)
        
        reset_state, reward, done = env.reset()
        self.model.eval()
        self.model.requires_grad_(False)
        self.model.pre_forward(reset_state)
        
        with torch.no_grad():
            policy_solutions, policy_prob, rewards = rollout(self.model, env, 'greedy')
        
        # 获取POMO最佳解
        aug_reward = rewards.reshape(self.aug_factor, 1, env.multi_width)
        max_pomo_reward, pomo_indices = aug_reward.max(dim=2)
        max_aug_pomo_reward, aug_indices = max_pomo_reward.max(dim=0)
        
        best_augmentation_idx = aug_indices.item()
        best_pomo_idx = pomo_indices[best_augmentation_idx, 0].item()
        stage1_solution = policy_solutions[best_augmentation_idx, best_pomo_idx, :]
        stage1_cost = -max_aug_pomo_reward.float().item()
        
        pomo_time = time.time() - pomo_start
        print(f"  [POMO阶段] 完成 (用时: {pomo_time:.2f}s, cost: {stage1_cost:.2f})")
        
        # 记录POMO阶段结果（迭代0）
        recorder.record_iteration(0, stage1_cost, stage1_solution, phase='POMO')
        
        # =========================
        # 第二阶段：KLH优化
        # =========================
        print("  [KLH阶段] 开始...")
        
        vrp_file = self.vrplib_path + '/' + name + '.vrp'
        
        # KLH优化，支持迭代记录
        final_solution, final_cost = self.stage2_lkh_with_iterations(
            vrp_file, stage1_solution, optimal, recorder
        )
        
        # 整理结果
        timepoint_results = recorder.get_results_at_timepoints()
        iteration_records = recorder.get_all_iterations()
        
        # 添加规模信息
        for tp in timepoint_results:
            timepoint_results[tp]['scale'] = problem_size
        
        return {
            'timepoint_results': timepoint_results,
            'iteration_records': iteration_records,
            'scale': problem_size,
            'optimal': optimal,
            'final_cost': final_cost
        }
    
    def stage2_lkh_with_iterations(self, vrp_file_path, stage1_solution, optimal, recorder):
        """第二阶段LKH优化，支持迭代记录"""
        from KLH import parse_stage1_solution
        
        # 解析第一阶段解
        routes_format = parse_stage1_solution(stage1_solution)
        
        # 创建LKH优化器
        optimizer = LKHOptimizer(distance_matrix=np.array([]), demands=[], vehicle_capacity=0)
        problem_data = optimizer.parse_vrp_file(vrp_file_path)
        distance_matrix = optimizer.calculate_distance_matrix(problem_data['coordinates'])
        
        optimizer.distance_matrix = distance_matrix
        optimizer.demands = problem_data['demands']
        optimizer.vehicle_capacity = problem_data['capacity']
        
        # 执行LKH优化（带迭代记录）
        optimized_routes, best_cost = self.lkh_optimize_with_iterations(
            optimizer, routes_format, optimal, recorder
        )
        
        optimized_solution = convert_solution_to_sequence(optimized_routes)
        return optimized_solution, best_cost
    
    def lkh_optimize_with_iterations(self, optimizer, initial_solution, optimal, recorder, max_iterations=100):
        """LKH优化循环，记录每次迭代，迭代结束后才检查时间"""
        current_solution = [route.copy() for route in initial_solution]
        best_solution = [route.copy() for route in initial_solution]
        best_cost = optimizer.calculate_total_cost(best_solution)
        
        no_improvement_count = 0
        max_no_improvement = 10000
        
        for iteration in range(1, max_iterations + 1):
            improved = False
            
            # 路径内优化
            for i in range(len(current_solution)):
                if len(current_solution[i]) > 0:
                    current_solution[i] = optimizer.two_opt_intra_route(current_solution[i])
                    current_solution[i] = optimizer.or_opt(current_solution[i])
            
            # 路径间优化
            for i in range(len(current_solution)):
                for j in range(i + 1, len(current_solution)):
                    if len(current_solution[i]) > 0 and len(current_solution[j]) > 0:
                        new_route1, new_route2 = optimizer.cross_exchange(
                            current_solution[i], current_solution[j])
                        current_solution[i] = new_route1
                        current_solution[j] = new_route2
            
            # 客户重定位
            current_solution = optimizer.relocate_customer(current_solution)
            
            # 计算当前成本
            current_cost = optimizer.calculate_total_cost(current_solution)
            
            # 检查是否有改进
            if current_cost < best_cost:
                best_solution = [route.copy() for route in current_solution]
                best_cost = current_cost
                improved = True
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            # 记录当前迭代结果
            recorder.record_iteration(iteration, best_cost, best_solution, phase='KLH')
            
            # ====== 关键修改：迭代结束后才检查时间 ======
            # 检查是否超过最大时间（在迭代完成后检查）
            if recorder.start_time:
                elapsed = time.time() - recorder.start_time
                if elapsed > recorder.target_times[-1]:
                    print(f"  迭代{iteration}完成后，已超过最大时间限制 {recorder.target_times[-1]}s（实际{elapsed:.2f}s），停止优化")
                    break
            
            # 检查最优解
            if optimal is not None and best_cost <= optimal:
                print(f"  达到最优解! cost={best_cost:.2f} <= optimal={optimal}")
                break
            
            # 早停
            if no_improvement_count >= max_no_improvement:
                print(f"  连续{max_no_improvement}次无改进，提前终止")
                break
        
        print(f"  [KLH阶段] 完成 (最终cost: {best_cost:.2f}, 共{iteration}次迭代)")
        return best_solution, best_cost
    
    def save_instance_results(self, instance_name, results, output_dir='results/instances'):
        """保存单个实例的详细结果到独立Excel文件"""
        os.makedirs(output_dir, exist_ok=True)
        excel_file = os.path.join(output_dir, f'{instance_name}_iterations.xlsx')
        
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            # Sheet 1: 迭代详情
            df_iterations = pd.DataFrame(results['iteration_records'])
            df_iterations.to_excel(writer, sheet_name='Iterations', index=False)
            
            # Sheet 2: 时间点汇总
            timepoint_data = []
            for tp in sorted(self.time_points):
                if tp in results['timepoint_results']:
                    res = results['timepoint_results'][tp]
                    timepoint_data.append({
                        'Time_Limit': tp,
                        'Cost': res['cost'],
                        'Gap': res['gap'],
                        'Actual_Time': res['actual_time'],
                        'Iteration': res['iteration']
                    })
            
            if timepoint_data:
                df_timepoints = pd.DataFrame(timepoint_data)
                df_timepoints.to_excel(writer, sheet_name='Timepoints', index=False)
            
            # Sheet 3: 实例信息
            info_data = [{
                'Instance': instance_name,
                'Scale': results['scale'],
                'Optimal': results['optimal'],
                'Final_Cost': results['final_cost'],
                'Final_Gap': round((results['final_cost'] - results['optimal']) / results['optimal'] * 100, 2),
                'Total_Iterations': len(results['iteration_records']),
                'Total_Time': results['iteration_records'][-1]['time'] if results['iteration_records'] else 0
            }]
            df_info = pd.DataFrame(info_data)
            df_info.to_excel(writer, sheet_name='Info', index=False)
        
        print(f"  实例结果已保存: {excel_file}")
    
    def generate_summary(self):
        """生成汇总统计"""
        print(f"\n{'='*60}")
        print("汇总统计")
        print(f"{'='*60}")
        
        for tp in self.time_points:
            gaps = []
            for instance_name, results in self.all_results.items():
                if tp in results['timepoint_results']:
                    gaps.append(results['timepoint_results'][tp]['gap'])
            
            if gaps:
                avg_gap = np.mean(gaps)
                print(f"时间点 {tp}s: 平均Gap = {avg_gap:.2f}% ({len(gaps)}/{len(self.all_results)}个实例)")
    
    def export_summary_to_excel(self, output_dir='results'):
        """导出汇总结果到Excel"""
        os.makedirs(output_dir, exist_ok=True)
        excel_file = os.path.join(output_dir, 'pomo_klh_summary.xlsx')
        
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            # 工作表1: 按时间点汇总
            summary_data = []
            for tp in sorted(self.time_points):
                row = {'Time_Limit': tp}
                
                gaps = []
                costs = []
                for instance_name, results in self.all_results.items():
                    if tp in results['timepoint_results']:
                        gaps.append(results['timepoint_results'][tp]['gap'])
                        costs.append(results['timepoint_results'][tp]['cost'])
                
                if gaps:
                    row['Avg_Gap'] = round(np.mean(gaps), 2)
                    row['Std_Gap'] = round(np.std(gaps), 2)
                    row['Min_Gap'] = round(np.min(gaps), 2)
                    row['Max_Gap'] = round(np.max(gaps), 2)
                    row['Avg_Cost'] = round(np.mean(costs), 2)
                    row['Solved'] = f"{len(gaps)}/{len(self.all_results)}"
                else:
                    row['Avg_Gap'] = ''
                    row['Std_Gap'] = ''
                    row['Min_Gap'] = ''
                    row['Max_Gap'] = ''
                    row['Avg_Cost'] = ''
                    row['Solved'] = f"0/{len(self.all_results)}"
                
                summary_data.append(row)
            
            df_summary = pd.DataFrame(summary_data)
            df_summary.to_excel(writer, sheet_name='Timepoint_Summary', index=False)
            
            # 工作表2: 各实例在各时间点的gap
            gap_matrix_data = []
            for instance_name, results in self.all_results.items():
                row = {'Instance': instance_name}
                for tp in sorted(self.time_points):
                    if tp in results['timepoint_results']:
                        row[f'{tp}s'] = results['timepoint_results'][tp]['gap']
                    else:
                        row[f'{tp}s'] = ''
                gap_matrix_data.append(row)
            
            df_gap_matrix = pd.DataFrame(gap_matrix_data)
            df_gap_matrix.to_excel(writer, sheet_name='Gap_Matrix', index=False)
            
            # 工作表3: 实例信息汇总
            instance_info_data = []
            for instance_name, results in self.all_results.items():
                instance_info_data.append({
                    'Instance': instance_name,
                    'Scale': results['scale'],
                    'Optimal': results['optimal'],
                    'Final_Cost': results['final_cost'],
                    'Final_Gap': round((results['final_cost'] - results['optimal']) / results['optimal'] * 100, 2),
                    'Total_Iterations': len(results['iteration_records']),
                    'Total_Time': results['iteration_records'][-1]['time'] if results['iteration_records'] else 0
                })
            
            df_instance_info = pd.DataFrame(instance_info_data)
            df_instance_info.to_excel(writer, sheet_name='Instance_Info', index=False)
        
        print(f"\n{'='*60}")
        print(f"汇总Excel文件已保存: {excel_file}")
        print(f"各实例详细结果保存在: results/instances/ 目录")
        print(f"{'='*60}")


if __name__ == "__main__":
    
    # 定义目标目录路径
    target_directory = r"elg-master\cvrp"
    
    # 使用 os.chdir() 切换到目标目录
    os.chdir(target_directory)
    
    with open('config.yml', 'r', encoding='utf-8') as config_file:
        config = yaml.load(config_file.read(), Loader=yaml.FullLoader)
    
    # 自定义时间点
    time_points = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
    
    tester = VRPLib_Tester_WithIterations(config=config, time_points=time_points)
    tester.test_on_vrplib()