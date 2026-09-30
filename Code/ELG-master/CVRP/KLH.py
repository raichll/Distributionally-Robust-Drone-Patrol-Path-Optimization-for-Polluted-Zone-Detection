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
        LKH算法优化器（超轻量化版本 - 完全兼容版）
        
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
        self.n_customers = len(distance_matrix) - 1
        
        # 预计算最近邻居列表
        self.nearest_neighbors = {}
        self._precompute_nearest_neighbors(k=min(10, self.n_customers))
        
        # 成本缓存
        self.cost_cache = {}
        
    def _precompute_nearest_neighbors(self, k: int = 10):
        """预计算每个节点的k个最近邻居"""
        n = len(self.distance_matrix)
        for i in range(n):
            distances = [(j, self.distance_matrix[i][j]) for j in range(n) if j != i]
            distances.sort(key=lambda x: x[1])
            self.nearest_neighbors[i] = [node for node, _ in distances[:k]]
        
    def parse_vrp_file(self, filepath: str) -> Dict:
        """解析VRP文件"""
        problem_data = {}
        coordinates = {}
        demands = {}
        
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
        
        max_node = max(coordinates.keys())
        coord_list = []
        demand_list = []
        
        for i in range(1, max_node + 1):
            if i in coordinates:
                coord_list.append(coordinates[i])
                demand_list.append(demands.get(i, 0))
            else:
                coord_list.append((0, 0))
                demand_list.append(0)
        
        problem_data['coordinates'] = coord_list
        problem_data['demands'] = demand_list
        problem_data['original_node_mapping'] = coordinates
        return problem_data
         
    def calculate_distance_matrix(self, coordinates: List[Tuple[float, float]]) -> np.ndarray:
        """计算距离矩阵（使用PyTorch加速）"""
        coords_tensor = torch.FloatTensor(coordinates)
        dist_matrix = (coords_tensor[:, None, :] - coords_tensor[None, :, :]).norm(p=2, dim=-1)
        return dist_matrix.numpy()

    def calculate_route_cost(self, route: List[int]) -> float:
        """计算路径总成本（带缓存）"""
        if not route:
            return 0
        
        route_key = tuple(route)
        if route_key in self.cost_cache:
            return self.cost_cache[route_key]
        
        cost = 0
        current = self.depot
        
        for customer in route:
            if customer < len(self.distance_matrix):
                cost += self.distance_matrix[current][customer]
                current = customer
            else:
                return float('inf')
        
        if current < len(self.distance_matrix):
            cost += self.distance_matrix[current][self.depot]
        
        self.cost_cache[route_key] = cost
        return cost
    
    def calculate_route_cost_delta(self, route: List[int], i: int, j: int) -> float:
        """计算2-opt交换后的成本变化（增量计算）"""
        if i >= j or j > len(route):
            return 0
        
        prev_i = self.depot if i == 0 else route[i-1]
        next_j = self.depot if j == len(route) else route[j]
        
        old_cost = self.distance_matrix[prev_i][route[i]] + self.distance_matrix[route[j-1]][next_j]
        new_cost = self.distance_matrix[prev_i][route[j-1]] + self.distance_matrix[route[i]][next_j]
        
        return new_cost - old_cost
    
    def is_feasible_route(self, route: List[int]) -> bool:
        """检查路径是否满足容量约束"""
        total_demand = 0
        for customer in route:
            if customer < len(self.demands):
                total_demand += self.demands[customer]
            else:
                return False
        return total_demand <= self.vehicle_capacity
    
    def calculate_total_cost(self, solution: List[List[int]]) -> float:
        """计算解的总成本"""
        return sum(self.calculate_route_cost(route) for route in solution)
    
    # ==================== 兼容原测试文件的方法名 ====================
    
    def two_opt_intra_route(self, route: List[int]) -> List[int]:
        """
        路径内2-opt优化（超快速版本）
        保持原方法名以兼容测试文件
        """
        if len(route) < 3:
            return route
        
        best_route = route.copy()
        best_improvement = 0
        best_i, best_j = -1, -1
        
        # 只扫描一次，记录最佳改进
        for i in range(len(route) - 1):
            for j in range(i + 2, min(i + 6, len(route) + 1)):
                delta = self.calculate_route_cost_delta(route, i, j)
                
                if delta < best_improvement - 1e-6:
                    best_improvement = delta
                    best_i, best_j = i, j
        
        # 应用最佳改进
        if best_i != -1:
            best_route = route[:best_i] + route[best_i:best_j][::-1] + route[best_j:]
            if tuple(route) in self.cost_cache:
                del self.cost_cache[tuple(route)]
        
        return best_route
    
    def or_opt(self, route: List[int]) -> List[int]:
        """
        Or-opt优化（简化版本）
        保持原方法名以兼容测试文件
        由于超轻量化，这里只做简单的客户重定位
        """
        if len(route) < 4:
            return route
        
        best_route = route.copy()
        best_cost = self.calculate_route_cost(best_route)
        
        # 只尝试移动1个客户，且只尝试3个位置
        for i in range(len(route)):
            customer = route[i]
            remaining = route[:i] + route[i+1:]
            
            # 只尝试3个位置：前、中、后
            positions = [0, len(remaining) // 2, len(remaining)]
            
            for j in positions:
                if j != i and j != i + 1:  # 避免无意义移动
                    new_route = remaining[:j] + [customer] + remaining[j:]
                    
                    if self.is_feasible_route(new_route):
                        new_cost = self.calculate_route_cost(new_route)
                        if new_cost < best_cost - 1e-6:
                            best_route = new_route
                            best_cost = new_cost
                            return best_route  # First improvement
        
        return best_route
    
    def cross_exchange(self, route1: List[int], route2: List[int]) -> Tuple[List[int], List[int]]:
        """
        路径间交叉交换（简化为客户交换）
        保持原方法名以兼容测试文件
        """
        if not route1 or not route2:
            return route1, route2
        
        best_route1, best_route2 = route1.copy(), route2.copy()
        best_cost = self.calculate_route_cost(route1) + self.calculate_route_cost(route2)
        
        # 只尝试5次随机交换
        for _ in range(5):
            i = random.randint(0, len(route1) - 1)
            j = random.randint(0, len(route2) - 1)
            
            new_route1 = route1.copy()
            new_route2 = route2.copy()
            new_route1[i], new_route2[j] = new_route2[j], new_route1[i]
            
            if self.is_feasible_route(new_route1) and self.is_feasible_route(new_route2):
                new_cost = self.calculate_route_cost(new_route1) + self.calculate_route_cost(new_route2)
                
                if new_cost < best_cost - 1e-6:
                    best_route1, best_route2 = new_route1, new_route2
                    best_cost = new_cost
                    return best_route1, best_route2  # First improvement
        
        return best_route1, best_route2
    
    def relocate_customer(self, solution: List[List[int]]) -> List[List[int]]:
        """
        客户重定位（超快速版本）
        保持原方法名以兼容测试文件
        """
        best_solution = [route.copy() for route in solution]
        best_cost = self.calculate_total_cost(best_solution)
        
        # 随机采样一些客户
        all_customers = [(i, j, customer) 
                        for i, route in enumerate(solution) 
                        for j, customer in enumerate(route)]
        
        if not all_customers:
            return best_solution
        
        # 只尝试3次重定位
        sample_size = min(len(all_customers), 9)
        sampled_customers = random.sample(all_customers, sample_size)
        
        relocations = 0
        for i, j, customer in sampled_customers:
            if relocations >= 3:
                break
            
            route1 = solution[i]
            
            # 只考虑2个候选路径
            candidate_routes = []
            for k in range(len(solution)):
                if k != i and len(solution[k]) > 0:
                    has_neighbor = any(node in self.nearest_neighbors.get(customer, []) 
                                     for node in solution[k])
                    if has_neighbor:
                        candidate_routes.append(k)
                        if len(candidate_routes) >= 2:
                            break
            
            if not candidate_routes:
                candidate_routes = [k for k in range(len(solution)) if k != i][:2]
            
            for k in candidate_routes:
                if relocations >= 3:
                    break
                
                route2 = solution[k]
                new_route1 = route1[:j] + route1[j+1:]
                
                # 只尝试3个位置
                positions = [0, len(route2) // 2, len(route2)]
                
                for pos in positions:
                    new_route2 = route2[:pos] + [customer] + route2[pos:]
                    
                    if self.is_feasible_route(new_route2):
                        temp_solution = [route.copy() for route in solution]
                        temp_solution[i] = new_route1
                        temp_solution[k] = new_route2
                        new_cost = self.calculate_total_cost(temp_solution)
                        
                        if new_cost < best_cost - 1e-6:
                            best_solution = temp_solution
                            best_cost = new_cost
                            relocations += 1
                            return best_solution  # First improvement
        
        return best_solution
    
    # ==================== 原有的lkh_optimize方法 ====================
    
    def lkh_optimize(self, initial_solution: List[List[int]], op, max_iterations: int = 100) -> Tuple[List[List[int]], float]:
        """
        LKH算法主循环（超轻量化版本）
        保持原方法签名以兼容测试文件
        """
        current_solution = [route.copy() for route in initial_solution]
        best_solution = [route.copy() for route in initial_solution]
        best_cost = self.calculate_total_cost(best_solution)
        
        print(f"初始解成本: {best_cost:.2f}")
        
        no_improvement_count = 0
        max_no_improvement = 3
        
        for iteration in range(max_iterations):
            improved = False
            
            # 1. 路径内优化：2-opt
            for i in range(len(current_solution)):
                if len(current_solution[i]) > 0:
                    current_solution[i] = self.two_opt_intra_route(current_solution[i])
                    
                    # Or-opt只在路径较短时执行
                    if len(current_solution[i]) < 20:
                        current_solution[i] = self.or_opt(current_solution[i])
            
            # 2. 路径间优化：客户交换
            num_routes = len(current_solution)
            if num_routes > 1:
                num_pairs = min(2, (num_routes * (num_routes - 1)) // 2)
                route_pairs = []
                
                for i in range(num_routes):
                    for j in range(i + 1, num_routes):
                        if len(current_solution[i]) > 0 and len(current_solution[j]) > 0:
                            route_pairs.append((i, j))
                
                if route_pairs:
                    selected_pairs = random.sample(route_pairs, min(num_pairs, len(route_pairs)))
                    
                    for i, j in selected_pairs:
                        new_route1, new_route2 = self.cross_exchange(
                            current_solution[i], current_solution[j])
                        current_solution[i] = new_route1
                        current_solution[j] = new_route2
            
            # 3. 客户重定位（每3次迭代执行一次）
            if iteration % 3 == 0:
                current_solution = self.relocate_customer(current_solution)
            
            # 检查是否有改进
            current_cost = self.calculate_total_cost(current_solution)

            if op is not None:    
                if current_cost < op:
                    best_solution = [route.copy() for route in current_solution]
                    best_cost = current_cost
                    print(f"迭代 {iteration}: 找到优于最优解的解 {best_cost:.2f} < {op}")
                    break

            if current_cost < best_cost - 1e-6:
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
        
        # 清空缓存
        self.cost_cache.clear()
        
        return best_solution, best_cost

def parse_stage1_solution(solution_sequence: List[int]) -> List[List[int]]:
    """解析第一阶段的解格式"""
    routes = []
    current_route = []
    
    for node in solution_sequence:
        if node == 0:
            if current_route:
                routes.append(current_route)
                current_route = []
        else:
            if 0 < node:
                current_route.append(node)
            else:
                print(f"警告：发现无效节点编号 {node}，已忽略")
    
    if current_route:
        routes.append(current_route)
    
    return routes

def convert_solution_to_sequence(solution: List[List[int]]) -> List[int]:
    """将路径列表格式转换回序列格式"""
    sequence = [0]
    
    for route in solution:
        if route:
            sequence.extend(route)
            sequence.append(0)
    
    return sequence

def stage2_lkh_optimization(vrp_file_path: str, stage1_solution: List[int], op) -> Tuple[List[int], float]:
    """第二阶段LKH优化主函数"""
    print("解析第一阶段解...")
    routes_format = parse_stage1_solution(stage1_solution)
    
    optimizer = LKHOptimizer(distance_matrix=np.array([]), demands=[], vehicle_capacity=0)
    
    problem_data = optimizer.parse_vrp_file(vrp_file_path)
    distance_matrix = optimizer.calculate_distance_matrix(problem_data['coordinates'])
    
    optimizer.distance_matrix = distance_matrix
    optimizer.demands = problem_data['demands']
    optimizer.vehicle_capacity = problem_data['capacity']
    
    print("开始第二阶段LKH优化...")
    print(f"问题规模: {len(problem_data['coordinates'])}个节点")
    print(f"车辆容量: {problem_data['capacity']}")
    print(f"初始解路径数: {len(routes_format)}")
    
    optimized_routes, best_cost = optimizer.lkh_optimize(routes_format, op)
    
    optimized_solution = convert_solution_to_sequence(optimized_routes)
        
    return optimized_solution, best_cost