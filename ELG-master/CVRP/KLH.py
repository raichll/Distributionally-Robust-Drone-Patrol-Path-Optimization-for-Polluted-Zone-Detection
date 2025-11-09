import numpy as np
import math
import random
from typing import List, Tuple, Dict
import copy
import torch
class LKHOptimizer:
    def __init__(self, distance_matrix: np.ndarray, demands: List[float], 
                 vehicle_capacity: float, depot: int = 0):
        """
        LKH算法优化器
        
        Args:
            distance_matrix: 距离矩阵
            demands: 客户需求量
            vehicle_capacity: 车辆容量
            depot: 配送中心索引
        """
        self.distance_matrix = distance_matrix
        self.demands = demands
        self.vehicle_capacity = vehicle_capacity
        self.depot = depot
        self.n_customers = len(distance_matrix) - 1  # 不包括depot
        
    def parse_vrp_file(self, filepath: str) -> Dict:
        """解析VRP文件"""
        problem_data = {}
        coordinates = {}  # 改为字典，保持原始节点编号
        demands = {}      # 改为字典，保持原始节点编号
        
        with open(filepath, 'r') as file:
            lines = file.readlines()
            
        reading_coords = False
        reading_demands = False
        
        for line in lines:
            line = line.strip()
            if line.startswith('CAPACITY'):
                problem_data['capacity'] = int(line.split(':')[1].strip())
            elif line.startswith('NODE_COORD_SECTION'):
                reading_coords = True
                continue
            elif line.startswith('DEMAND_SECTION'):
                reading_coords = False
                reading_demands = True
                continue
            elif line.startswith('DEPOT_SECTION'):
                break
            elif reading_coords and line and not line.startswith('EOF'):
                parts = line.split()
                if len(parts) >= 3:
                    node_id = int(parts[0])
                    coordinates[node_id] = (float(parts[1]), float(parts[2]))
            elif reading_demands and line and not line.startswith('EOF'):
                parts = line.split()
                if len(parts) >= 2:
                    node_id = int(parts[0])
                    demands[node_id] = float(parts[1])
        
        # 转换为0-based索引的列表格式
        max_node = max(coordinates.keys())
        coord_list = []
        demand_list = []
        
        for i in range(1, max_node + 1):  # 原始编号1到max_node
            if i in coordinates:
                coord_list.append(coordinates[i])
                demand_list.append(demands.get(i, 0))
            else:
                coord_list.append((0, 0))  # 默认坐标
                demand_list.append(0)      # 默认需求
        
        problem_data['coordinates'] = coord_list
        problem_data['demands'] = demand_list
        problem_data['original_node_mapping'] = coordinates
        return problem_data
    '''    
    def calculate_distance_matrix(self, coordinates: List[Tuple[float, float]]) -> np.ndarray:
        """计算距离矩阵"""
        n = len(coordinates)
        dist_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    dx = coordinates[i][0] - coordinates[j][0]
                    dy = coordinates[i][1] - coordinates[j][1]
                    dist_matrix[i][j] = math.sqrt(dx*dx + dy*dy)
        
        return dist_matrix
        '''       
    def calculate_distance_matrix(self,coordinates: List[Tuple[float, float]]) -> np.ndarray:
        
            # 将坐标转换为PyTorch张量
            coords_tensor = torch.FloatTensor(coordinates)
        # shape: (n, 2)
        
        # 按照原文方法计算距离矩阵
        # (coords_tensor[:, None, :] - coords_tensor[None, :, :]) 创建了所有点对之间的向量差
        # coords_tensor[:, None, :] shape: (n, 1, 2)
        # coords_tensor[None, :, :] shape: (1, n, 2)
        # 相减后得到 shape: (n, n, 2) 的张量，包含所有点对之间的向量差
        
            dist_matrix = (coords_tensor[:, None, :] - coords_tensor[None, :, :]).norm(p=2, dim=-1)
        # shape: (n, n)
        
        # 转换回NumPy数组
            return dist_matrix.numpy()


    def calculate_route_cost(self, route: List[int]) -> float:
        """计算路径总成本"""
        if not route:
            return 0
        
        cost = 0
        current = self.depot  # depot在距离矩阵中的索引是0
        
        for customer in route:
            # customer是1-100的编号，对应距离矩阵中的索引1-100
            # 但需要确保不超出矩阵范围
            if customer < len(self.distance_matrix):
                cost += self.distance_matrix[current][customer]
                current = customer
            else:
                print(f"警告：客户编号 {customer} 超出距离矩阵范围")
                return float('inf')  # 返回无穷大表示无效路径
        
        # 回到depot
        if current < len(self.distance_matrix):
            cost += self.distance_matrix[current][self.depot]
        
        return cost
    
    def is_feasible_route(self, route: List[int]) -> bool:
        """检查路径是否满足容量约束"""
        total_demand = 0
        for customer in route:
            # customer是1-100的编号，对应demands列表中的索引1-100
            if customer < len(self.demands):
                total_demand += self.demands[customer]
            else:
                print(f"警告：客户编号 {customer} 超出需求数组范围")
                return False
        return total_demand <= self.vehicle_capacity
    
    def calculate_total_cost(self, solution: List[List[int]]) -> float:
        """计算解的总成本"""
        return sum(self.calculate_route_cost(route) for route in solution)
    
    def two_opt_intra_route(self, route: List[int]) -> List[int]:
        """路径内2-opt优化"""
        if len(route) < 3:
            return route
        
        best_route = route.copy()
        best_cost = self.calculate_route_cost(best_route)
        improved = True
        
        while improved:
            improved = False
            for i in range(len(route) - 1):
                for j in range(i + 2, len(route)):
                    # 创建新路径
                    new_route = route[:i+1] + route[i+1:j+1][::-1] + route[j+1:]
                    
                    if self.is_feasible_route(new_route):
                        new_cost = self.calculate_route_cost(new_route)
                        if new_cost < best_cost:
                            best_route = new_route
                            best_cost = new_cost
                            route = new_route
                            improved = True
        
        return best_route
    
    def or_opt(self, route: List[int]) -> List[int]:
        """Or-opt优化（移动1-3个连续客户到其他位置）"""
        if len(route) < 4:
            return route
        
        best_route = route.copy()
        best_cost = self.calculate_route_cost(best_route)
        
        for segment_length in [1, 2, 3]:
            for i in range(len(route) - segment_length + 1):
                segment = route[i:i+segment_length]
                remaining = route[:i] + route[i+segment_length:]
                
                for j in range(len(remaining) + 1):
                    new_route = remaining[:j] + segment + remaining[j:]
                    
                    if self.is_feasible_route(new_route):
                        new_cost = self.calculate_route_cost(new_route)
                        if new_cost < best_cost:
                            best_route = new_route
                            best_cost = new_cost
        
        return best_route
    
    def cross_exchange(self, route1: List[int], route2: List[int]) -> Tuple[List[int], List[int]]:
        """路径间交叉交换"""
        best_route1, best_route2 = route1.copy(), route2.copy()
        best_cost = self.calculate_route_cost(route1) + self.calculate_route_cost(route2)
        
        for i in range(len(route1)):
            for j in range(len(route2)):
                for len1 in range(1, min(4, len(route1) - i + 1)):
                    for len2 in range(1, min(4, len(route2) - j + 1)):
                        # 交换片段
                        segment1 = route1[i:i+len1]
                        segment2 = route2[j:j+len2]
                        
                        new_route1 = route1[:i] + segment2 + route1[i+len1:]
                        new_route2 = route2[:j] + segment1 + route2[j+len2:]
                        
                        if (self.is_feasible_route(new_route1) and 
                            self.is_feasible_route(new_route2)):
                            new_cost = (self.calculate_route_cost(new_route1) + 
                                       self.calculate_route_cost(new_route2))
                            
                            if new_cost < best_cost:
                                best_route1, best_route2 = new_route1, new_route2
                                best_cost = new_cost
        
        return best_route1, best_route2
    
    def relocate_customer(self, solution: List[List[int]]) -> List[List[int]]:
        """客户重定位"""
        best_solution = [route.copy() for route in solution]
        best_cost = self.calculate_total_cost(best_solution)
        
        for i, route1 in enumerate(solution):
            for j, customer in enumerate(route1):
                # 尝试将客户移动到其他路径
                for k, route2 in enumerate(solution):
                    if i == k:
                        continue
                    
                    # 从route1移除customer
                    new_route1 = route1[:j] + route1[j+1:]
                    
                    # 尝试插入到route2的每个位置
                    for pos in range(len(route2) + 1):
                        new_route2 = route2[:pos] + [customer] + route2[pos:]
                        
                        if self.is_feasible_route(new_route2):
                            new_solution = best_solution.copy()
                            new_solution[i] = new_route1
                            new_solution[k] = new_route2
                            
                            new_cost = self.calculate_total_cost(new_solution)
                            if new_cost < best_cost:
                                best_solution = new_solution
                                best_cost = new_cost
        
        return best_solution
    
    def lkh_optimize(self, initial_solution: List[List[int]], op, max_iterations: int = 100) -> List[List[int]]:
        """
        LKH算法主循环
        
        Args:
            initial_solution: 第一阶段强化学习得到的初始解
            max_iterations: 最大迭代次数
        """
        current_solution = [route.copy() for route in initial_solution]
        best_solution = [route.copy() for route in initial_solution]
        best_cost = self.calculate_total_cost(best_solution)
        
        print(f"初始解成本: {best_cost:.2f}")
        
        no_improvement_count = 0
        max_no_improvement = 3
        
        for iteration in range(max_iterations):
            improved = False
            
            # 1. 路径内优化
            for i in range(len(current_solution)):
                if len(current_solution[i]) > 0:
                    # 2-opt优化
                    optimized_route = self.two_opt_intra_route(current_solution[i])
                    current_solution[i] = optimized_route
                    
                    # Or-opt优化
                    optimized_route = self.or_opt(current_solution[i])
                    current_solution[i] = optimized_route
            
            # 2. 路径间优化
            for i in range(len(current_solution)):
                for j in range(i + 1, len(current_solution)):
                    if len(current_solution[i]) > 0 and len(current_solution[j]) > 0:
                        new_route1, new_route2 = self.cross_exchange(
                            current_solution[i], current_solution[j])
                        current_solution[i] = new_route1
                        current_solution[j] = new_route2
            
            # 3. 客户重定位
            current_solution = self.relocate_customer(current_solution)
            
            # 检查是否有改进
            current_cost = self.calculate_total_cost(current_solution)

            if op is not None:    
                if current_cost < op:
                    break

            if current_cost < best_cost:
                best_solution = [route.copy() for route in current_solution]
                best_cost = current_cost
                improved = True
                no_improvement_count = 0
                print(f"迭代 {iteration}: 新最优解成本 {best_cost:.2f}")
            else:
                no_improvement_count += 1
            
            # 早停条件
            if no_improvement_count >= max_no_improvement:
                print(f"连续{max_no_improvement}次无改进，提前终止")
                break
        
        print(f"LKH优化完成，最终成本: {best_cost:.2f}")
        return best_solution , best_cost

def parse_stage1_solution(solution_sequence: List[int]) -> List[List[int]]:
    """
    解析第一阶段的解格式，确保节点编号在有效范围内
    
    Args:
        solution_sequence: 格式如 [0,1,3,6,7,0,2,4,8,0] 的解序列
    
    Returns:
        转换后的路径列表格式 [[1,3,6,7], [2,4,8]]
    """
    routes = []
    current_route = []
    
    for node in solution_sequence:
        if node == 0:  # depot
            if current_route:  # 如果当前路径不为空
                routes.append(current_route)
                current_route = []
        else:
            # 确保节点编号在有效范围内（1-100，因为原始文件有101个节点，depot是1，客户是2-101）
            # 在解序列中：depot=0，客户1-100对应原始文件的客户2-101
            if 0 < node :
                current_route.append(node)
            else:
                print(f"警告：发现无效节点编号 {node}，已忽略")
    
    # 处理最后一个路径（如果序列不以0结尾）
    if current_route:
        routes.append(current_route)
    
    return routes

def convert_solution_to_sequence(solution: List[List[int]]) -> List[int]:
    """
    将路径列表格式转换回序列格式
    
    Args:
        solution: 路径列表格式 [[1,3,6,7], [2,4,8]]
    
    Returns:
        序列格式 [0,1,3,6,7,0,2,4,8,0]
    """
    sequence = [0]  # 从depot开始
    
    for route in solution:
        if route:  # 如果路径不为空
            sequence.extend(route)
            sequence.append(0)  # 回到depot
    
    return sequence

def stage2_lkh_optimization(vrp_file_path: str, stage1_solution: List[int],op) -> List[int]:
    """
    第二阶段LKH优化主函数
    
    Args:
        vrp_file_path: VRP问题文件路径
        stage1_solution: 第一阶段强化学习得到的解，格式如 [0,1,3,6,7,0,2,4,8,0]
    
    Returns:
        优化后的解（相同格式）
    """
    # 解析第一阶段解格式
    print("解析第一阶段解...")
    #print(f"原始解: {stage1_solution}")
    routes_format = parse_stage1_solution(stage1_solution)
    #print(f"转换后路径: {routes_format}")
    
    # 创建LKH优化器实例
    optimizer = LKHOptimizer(distance_matrix=np.array([]), demands=[], vehicle_capacity=0)
    
    # 解析VRP文件
    problem_data = optimizer.parse_vrp_file(vrp_file_path)
    
    # 计算距离矩阵
    distance_matrix = optimizer.calculate_distance_matrix(problem_data['coordinates'])
    
    # 更新优化器参数
    optimizer.distance_matrix = distance_matrix
    optimizer.demands = problem_data['demands']
    optimizer.vehicle_capacity = problem_data['capacity']
    
    print("开始第二阶段LKH优化...")
    print(f"问题规模: {len(problem_data['coordinates'])}个节点")
    print(f"车辆容量: {problem_data['capacity']}")
    print(f"初始解路径数: {len(routes_format)}")
    
    # 执行LKH优化
    optimized_routes,best_cost = optimizer.lkh_optimize(routes_format,op)
    
    # 转换回原始序列格式
    optimized_solution = convert_solution_to_sequence(optimized_routes)
        
    return optimized_solution, best_cost

  