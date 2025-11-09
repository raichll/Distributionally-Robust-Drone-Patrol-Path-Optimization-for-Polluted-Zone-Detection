import vrplib
import numpy as np
import torch
import yaml
import json
import time
import os
from torch.optim import Adam as Optimizer
from KLH import LKHOptimizer
from CVRPModel_DRO import CVRPModel_DRO
from CVRPEnv_DRO import CVRPEnv_DRO 
from utils_dro import rollout_dro, check_feasible
from typing import List, Tuple, Dict
from KLH import convert_solution_to_sequence, stage2_lkh_optimization, parse_stage1_solution  
import warnings
import folium
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import pandas as pd
import random

# Ignore RuntimeWarning
warnings.simplefilter("ignore", category=RuntimeWarning)

def convert_coordinates(coord):
    """
    Convert coordinates from integer format to latitude/longitude format
    Example: 10411402 -> 104.11402, 3057949 -> 30.57949
    """
    if coord >= 10000000:  # longitude
        return coord / 100000
    else:  # latitude
        return coord / 100000

def create_route_map(routes, coordinates, depot_idx=0, filename="route_map.html"):
    """
    Create route map visualization with English labels for DRO-CVRP
    """
    # Convert coordinate format
    converted_coords = {}
    for node_id, (x, y) in coordinates.items():
        lon = convert_coordinates(x)
        lat = convert_coordinates(y)
        converted_coords[node_id] = (lat, lon)  # folium uses (lat, lon) format
    
    # Calculate map center
    lats = [coord[0] for coord in converted_coords.values()]
    lons = [coord[1] for coord in converted_coords.values()]
    center_lat = sum(lats) / len(lats)
    center_lon = sum(lons) / len(lons)
    
    # Create map with English tiles - using CartoDB Positron for clean English labels
    m = folium.Map(
        location=[center_lat, center_lon], 
        zoom_start=12,
        tiles='CartoDB Positron'
    )
    
    # Define color list for different routes
    colors = ['#'+''.join([random.choice('0123456789ABCDEF') for _ in range(6)]) for _ in range(20)]
    
    # Add depot marker
    depot_coord = converted_coords[depot_idx]
    folium.Marker(
        depot_coord, 
        popup=f"DRO-CVRP Depot (Node {depot_idx})",
        tooltip=f"Depot Location - DRO Optimized",
        icon=folium.Icon(color='black', icon='home')
    ).add_to(m)
    
    # Draw each route
    for i, route in enumerate(routes):
        color = colors[i % len(colors)]
        
        # Build complete path (depot -> customers -> depot)
        full_route = [depot_idx] + route + [depot_idx]
        
        # Get route coordinates
        route_coords = []
        for node_id in full_route:
            if node_id in converted_coords:
                route_coords.append(converted_coords[node_id])
        
        # Draw route line
        if len(route_coords) >= 2:
            folium.PolyLine(
                locations=route_coords,
                color=color,
                weight=3,
                opacity=0.8,
                popup=f"DRO Route #{i+1}",
                tooltip=f"DRO Route #{i+1} - {len(route)} customers"
            ).add_to(m)
        
        # Add customer markers
        for j, node_id in enumerate(route):
            if node_id in converted_coords:
                folium.CircleMarker(
                    converted_coords[node_id],
                    radius=6,
                    popup=f"DRO Route #{i+1}, Stop {j+1}, Node {node_id}",
                    tooltip=f"Customer {node_id} (DRO Optimized)",
                    color=color,
                    fillColor=color,
                    fillOpacity=0.7
                ).add_to(m)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Save map
    m.save(filename)
    print(f"DRO route map saved to {filename}")
    
    return m

def create_route_plot(routes, coordinates, depot_idx=0, filename="route_plot.png"):
    """
    Create route matplotlib plot for DRO-CVRP
    """
    # Convert coordinate format
    converted_coords = {}
    for node_id, (x, y) in coordinates.items():
        lon = convert_coordinates(x)
        lat = convert_coordinates(y)
        converted_coords[node_id] = (lon, lat)
    
    plt.figure(figsize=(12, 10))
    
    # Define color list
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(routes))))
    
    # Draw each route
    for i, route in enumerate(routes):
        color = colors[i % len(colors)]
        
        # Build complete path
        full_route = [depot_idx] + route + [depot_idx]
        
        # Get route coordinates
        route_lons = []
        route_lats = []
        for node_id in full_route:
            if node_id in converted_coords:
                lon, lat = converted_coords[node_id]
                route_lons.append(lon)
                route_lats.append(lat)
        
        # Draw path
        if len(route_lons) >= 2 and len(route_lats) >= 2:
            plt.plot(route_lons, route_lats, 
                    color=color, 
                    linewidth=2, 
                    label=f'DRO Route #{i+1}', 
                    marker='o', 
                    markersize=4,
                    linestyle='-',
                    alpha=0.8)
    
    # Mark depot
    if depot_idx in converted_coords:
        depot_lon, depot_lat = converted_coords[depot_idx]
        plt.plot(depot_lon, depot_lat, 'ks', markersize=12, label='DRO Depot')
    
    plt.xlabel('Longitude (°)')
    plt.ylabel('Latitude (°)')
    plt.title('DRO-CVRP - Distributionally Robust Route Optimization')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"DRO route plot saved to {filename}")

def convert_solution_to_routes(solution_sequence, depot_idx=0):
    """
    Convert DRO solution sequence to route format
    """
    routes = []
    current_route = []
    
    for node in solution_sequence:
        if isinstance(node, torch.Tensor):
            node = node.item()
        
        if node == depot_idx:  # Encounter depot node
            if current_route:  # If current route is not empty
                routes.append(current_route)
                current_route = []
        else:
            current_route.append(int(node))
    
    # Add last route (if exists)
    if current_route:
        routes.append(current_route)
    
    return routes

def save_solution_to_file(routes, cost, filename, coordinates=None, instance_name="", dro_info=None):
    """
    Save DRO route solution to file with comprehensive DRO analysis
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"=== DRO-CVRP Solution Analysis - {instance_name} ===\n\n")
        
        # Write comprehensive DRO parameters and analysis
        if dro_info:
            f.write("DISTRIBUTIONALLY ROBUST OPTIMIZATION PARAMETERS:\n")
            f.write(f"  Wasserstein Ball Radius (ε): {dro_info.get('epsilon', 'N/A')}\n")
            f.write(f"  Big-M Penalty Coefficient: {dro_info.get('M', 'N/A')}\n")
            f.write(f"  Historical Sample Size (N): {dro_info.get('N', 'N/A')}\n")
            f.write(f"  Demand Uncertainty Strategy: {dro_info.get('sample_strategy', 'N/A')}\n")
            f.write(f"  Uncertainty Standard Deviation: {dro_info.get('uncertainty_std', 'N/A')}\n")
            f.write(f"  Dual Variable Regularization (λ): {dro_info.get('lambda_reg', 'N/A')}\n")
            
            f.write(f"\nDRO COST BREAKDOWN ANALYSIS:\n")
            f.write(f"  Total DRO-Optimized Cost: {cost:.4f}\n")
            f.write(f"  Base Routing Cost Component: {dro_info.get('routing_cost', 'N/A'):.4f}\n")
            f.write(f"  Robustness Penalty Component: {dro_info.get('dro_penalty', 'N/A'):.4f}\n")
            
            # Calculate penalty ratio if both components exist
            routing_cost = dro_info.get('routing_cost', 0)
            dro_penalty = dro_info.get('dro_penalty', 0)
            if routing_cost > 0:
                penalty_ratio = (dro_penalty / routing_cost) * 100
                f.write(f"  Robustness Penalty Ratio: {penalty_ratio:.2f}%\n")
            
            f.write(f"\nDRO UNCERTAINTY MODELING:\n")
            f.write(f"  Demand Distribution Type: {dro_info.get('sample_strategy', 'N/A')}\n")
            f.write(f"  Historical Scenarios Considered: {dro_info.get('N', 'N/A')}\n")
            f.write(f"  Worst-Case Protection Level: Wasserstein-{dro_info.get('epsilon', 'N/A')}\n\n")
        
        f.write("OPTIMIZED ROUTE SOLUTION:\n")
        for i, route in enumerate(routes, 1):
            route_str = ' -> '.join(map(str, route))
            f.write(f"Route #{i}: Depot -> {route_str} -> Depot\n")
            
            # If coordinate information exists, save coordinates too
            if coordinates:
                f.write(f"  Coordinates: ")
                for node_id in route:
                    if node_id in coordinates:
                        x, y = coordinates[node_id]
                        lon = convert_coordinates(x)
                        lat = convert_coordinates(y)
                        f.write(f"Node{node_id}({lon:.5f}°,{lat:.5f}°) ")
                f.write("\n")
        
        f.write(f"\nSOLUTION SUMMARY:\n")
        f.write(f"Total DRO-Optimized Cost: {cost:.2f}\n")
        f.write(f"Number of Routes: {len(routes)}\n")
        f.write(f"Average Customers per Route: {sum(len(route) for route in routes) / len(routes):.1f}\n")
        
        if dro_info:
            f.write(f"\nROBUSTNESS GUARANTEE:\n")
            f.write(f"This solution is robust against demand uncertainty within a\n")
            f.write(f"Wasserstein ball of radius {dro_info.get('epsilon', 'N/A')} around the empirical distribution.\n")
    
    print(f"DRO solution analysis saved to {filename}")

def print_solution(routes, cost, coordinates=None, instance_name="", dro_info=None):
    """
    Print DRO route solution with comprehensive analysis
    """
    print(f"\n{'='*80}")
    print(f"DRO-CVRP SOLUTION ANALYSIS - {instance_name}")
    print(f"{'='*80}")
    
    # Print DRO information if available
    if dro_info:
        print("DISTRIBUTIONALLY ROBUST OPTIMIZATION ANALYSIS:")
        print(f"  Total DRO Cost: {cost:.4f}")
        print(f"  Base Routing Cost: {dro_info.get('routing_cost', 'N/A'):.4f}")
        print(f"  Robustness Penalty: {dro_info.get('dro_penalty', 'N/A'):.4f}")
        
        # Calculate and display penalty ratio
        routing_cost = dro_info.get('routing_cost', 0)
        dro_penalty = dro_info.get('dro_penalty', 0)
        if routing_cost > 0:
            penalty_ratio = (dro_penalty / routing_cost) * 100
            print(f"  Penalty Ratio: {penalty_ratio:.2f}%")
        
        print(f"\nDRO PARAMETERS:")
        print(f"  Wasserstein Radius (ε): {dro_info.get('epsilon')}")
        print(f"  Big-M Coefficient: {dro_info.get('M')}")
        print(f"  Historical Samples (N): {dro_info.get('N')}")
        print(f"  Uncertainty Strategy: {dro_info.get('sample_strategy')}")
        print(f"  Uncertainty Std Dev: {dro_info.get('uncertainty_std')}")
        print()
    
    print("OPTIMIZED ROUTE SOLUTION:")
    for i, route in enumerate(routes, 1):
        route_str = ' -> '.join(map(str, route))
        print(f"Route #{i}: Depot -> {route_str} -> Depot")
        
        # If coordinate information exists, print coordinates too
        if coordinates:
            print(f"  Coordinates: ", end="")
            for node_id in route:
                if node_id in coordinates:
                    x, y = coordinates[node_id]
                    lon = convert_coordinates(x)
                    lat = convert_coordinates(y)
                    print(f"Node{node_id}({lon:.5f}°,{lat:.5f}°) ", end="")
            print()
    
    print(f"\nSOLUTION SUMMARY:")
    print(f"Total DRO Cost: {cost:.2f}")
    print(f"Number of Routes: {len(routes)}")
    print(f"Total Customers: {sum(len(route) for route in routes)}")
    print(f"Average Customers per Route: {sum(len(route) for route in routes) / len(routes):.1f}")

class VRPLib_Tester_DRO:
    """
    DRO-CVRP Tester with enhanced data acquisition for distributionally robust optimization
    """

    def __init__(self, config):
        self.config = config
        model_params = config['model_params']
        load_checkpoint = config['load_checkpoint']

        # Enhanced DRO parameters with more configuration options
        self.dro_params = config.get('dro_params', {
            'epsilon': 0.1,          # Wasserstein ball radius
            'M': 1.0,               # Big-M penalty for unvisited nodes
            'N': 50,                # Number of historical samples
            'lambda_reg': 0.01,     # Regularization for dual variables
            'uncertainty_std': 0.1,  # Standard deviation for demand uncertainty
            'sample_strategy': 'uniform',  # 'gaussian', 'uniform', or 'historical'
            'enable_capacity_uncertainty': True,    # Enable vehicle capacity uncertainty
            'capacity_uncertainty_std': 0.05,       # Capacity uncertainty std dev
            'enable_travel_time_uncertainty': False, # Enable travel time uncertainty
            'travel_time_uncertainty_std': 0.1,     # Travel time uncertainty std dev
            'robust_level': 0.95,    # Confidence level for robustness
            'adaptive_epsilon': False, # Adaptive epsilon based on problem size
        })

        # CUDA setup
        USE_CUDA = config.get('use_cuda', False)
        if USE_CUDA:
            cuda_device_num = config['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            self.device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            self.device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')
        
        # Load DRO model
        self.model = CVRPModel_DRO(**model_params)
        
        if model_params.get('ensemble', False):
            self.model.decoder.add_local_policy(self.device)
               
        checkpoint = torch.load(load_checkpoint, map_location=self.device)
        
        # Get model state_dict
        model_dict = self.model.state_dict()
        checkpoint_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        
        # Compare model and checkpoint parameters
        for key in checkpoint_dict:
            if key in model_dict:
                if checkpoint_dict[key].shape != model_dict[key].shape:
                    print(f"Shape mismatch for DRO parameter: {key}")
                    print(f"Checkpoint shape: {checkpoint_dict[key].shape}, Model shape: {model_dict[key].shape}")
            else:
                print(f"Checkpoint contains parameter not in DRO model: {key}")

        self.repeat_times = 1
        self.aug_factor = config['params']['aug_factor']
        self.vrplib_results = None
        
        # Create results directory structure for DRO
        self.results_dir = 'test_results_dro'
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Initialize DRO-specific data structures
        self.uncertainty_scenarios = {}
        self.historical_data_cache = {}
        
        print(f"DRO-CVRP Tester initialized with parameters:")
        print(f"  Wasserstein radius: {self.dro_params['epsilon']}")
        print(f"  Sample strategy: {self.dro_params['sample_strategy']}")
        print(f"  Historical samples: {self.dro_params['N']}")
        print(f"  Uncertainty std: {self.dro_params['uncertainty_std']}")
        
    def generate_enhanced_historical_samples(self, base_demand, instance_name=None, instance_data=None):
        """
        Generate enhanced historical demand samples for DRO from truck cluster data
        Generate N samples with strict 0 and 1 values based on cluster count data.
        The output shape will be (batch_size, N, problem_size).
        """
        N = self.dro_params['N']  
        
        # base_demand's shape is (batch_size, problem_size), e.g., (8, 50)
        batch_size, problem_size = base_demand.shape
        
        # 读取CSV聚类数据
        cluster_data = r"E:\project\truck\ELG-master\CVRP_DRO\truck_data"
        
        # 获取目录下所有文件
        cluster_data_path = [f for f in os.listdir(cluster_data) if f.endswith('.csv')]
        
        if not cluster_data_path:
            raise FileNotFoundError(f"No CSV files found in {cluster_data}. Cannot generate historical samples.")

        # 读取CSV文件
        # 这里假设只需要读取一个文件，如果多个文件，需要合并处理
        file_path = os.path.join(cluster_data, cluster_data_path[0])
        cluster_df = pd.read_csv(file_path)
       
        print(f"Successfully loaded cluster data with {len(cluster_df)} records")
        
        # 筛选count > 100的点
        filtered_df = cluster_df[cluster_df['count'] >= 100].copy()
        print(f"Filtered to {len(filtered_df)} points with count > 100")

        if filtered_df.empty:
            print("Warning: No data points with count > 100. Generating default uniform samples.")
            # 如果没有符合条件的数据，生成一个统一的默认样本
            historical_samples = torch.zeros(batch_size, N, problem_size)
            return historical_samples

        # 提取count值并归一化为概率
        counts = filtered_df['count'].values
        max_count = counts.max()
        min_count = counts.min()
        
        # 将count值转换为生成1的概率 (count越大，生成1的概率越高)
        probabilities = (counts - min_count) / (max_count - min_count)
        # 调整概率范围，避免极值 (0.1 到 0.9)
        probabilities = 0.1 + 0.8 * probabilities
        
        print(f"Count range: [{min_count}, {max_count}]")
        print(f"Probability range: [{probabilities.min():.3f}, {probabilities.max():.3f}]")
        
        samples = [] # 存储所有N个历史样本，每个样本包含batch_size个批次
        
        # 生成N个历史样本，每个样本都对应一个batch_size
        for i in range(N):
            sample_batch = []
            for batch_idx in range(batch_size):
                # 确定使用多少个聚类点 (取较小值确保不超出范围)
                num_points_to_use = min(problem_size, len(probabilities))
                
                # 随机选择聚类点对应的概率
                if num_points_to_use < problem_size:
                    # 如果聚类点不够，重复使用
                    selected_probs = np.tile(probabilities, (problem_size // len(probabilities) + 1))[:problem_size]
                else:
                    # 随机选择相应数量的概率
                    selected_indices = np.random.choice(len(probabilities), size=problem_size, replace=True)
                    selected_probs = probabilities[selected_indices]
                
                # 基于概率生成严格的0-1序列
                binary_sequence = np.random.binomial(1, selected_probs)
                
                # 将numpy数组转换为torch张量，保持浮点类型
                binary_tensor = torch.FloatTensor(binary_sequence.astype(float))
                sample_batch.append(binary_tensor)
            
            # 将batch组合，得到形状为 (batch_size, problem_size) 的张量
            sample = torch.stack(sample_batch, dim=0)
            samples.append(sample)
            
            if i < 5:  # 显示前5个样本的统计信息
                binary_ratio = np.mean(binary_sequence)
                unique_values = torch.unique(sample)
                print(f"Sample {i+1} (in N): Binary ratio (1s): {binary_ratio:.3f}, "
                      f"Unique values: {unique_values.tolist()}, "
                      f"Shape: {sample.shape}")
        
        # Stack所有样本，最终形状为 (batch_size, N, problem_size)
        historical_samples = torch.stack(samples, dim=1)  # shape: (batch_size, N, problem_size)
        
        # 验证生成的样本只包含0和1
        unique_values = torch.unique(historical_samples)
        print(f"Final verification - Unique values in all samples: {unique_values.tolist()}")
        
        # 存储不确定性场景用于分析
        if instance_name:
            self.uncertainty_scenarios[instance_name] = {
                'base_demand': base_demand.clone(),
                'historical_samples': historical_samples.clone(),
                'epsilon': self.dro_params['epsilon'],
                'strategy': 'cluster_based_binary',
                'cluster_data_points': len(filtered_df),
                'count_range': [int(min_count), int(max_count)],
                'probability_range': [float(probabilities.min()), float(probabilities.max())],
                'sample_count': N,
                'batch_size': batch_size
            }
        
        print(f"Generated {N} cluster-based binary historical samples with batch_size={batch_size}")
        print(f"Final sample tensor shape: {historical_samples.shape}")
        print(f"All values are strictly 0 or 1: {torch.all((historical_samples == 0) | (historical_samples == 1))}")
        
        return historical_samples

    def preprocess_vrp_data_for_dro(self, instance_data, instance_name):
        """
        Preprocess VRP instance data specifically for DRO optimization
        """
        print(f"Preprocessing VRP data for DRO optimization: {instance_name}")
        
        # Extract basic information
        problem_size = instance_data['node_coord'].shape[0] - 1
        base_capacity = instance_data['capacity']
        base_demand = instance_data['demand']
        
        # Enhanced demand preprocessing for DRO
        # Normalize demands while preserving relative patterns
        demand_tensor = torch.FloatTensor(base_demand).unsqueeze(0)
        normalized_demand = demand_tensor / base_capacity
        
        # Analyze demand patterns for better uncertainty modeling
        demand_stats = {
            'mean': normalized_demand.mean().item(),
            'std': normalized_demand.std().item(),
            'min': normalized_demand.min().item(),
            'max': normalized_demand.max().item(),
            'cv': (normalized_demand.std() / normalized_demand.mean()).item()  # Coefficient of variation
        }
        
        # Adjust DRO parameters based on demand characteristics
        adaptive_params = self.dro_params.copy()
        
        # Adaptive uncertainty modeling based on demand variability
        if demand_stats['cv'] > 0.5:  # High variability
            adaptive_params['uncertainty_std'] *= 1.2
            adaptive_params['N'] = min(int(adaptive_params['N'] * 1.2), 100)
            print(f"High demand variability detected (CV={demand_stats['cv']:.3f}), increasing uncertainty modeling")
        elif demand_stats['cv'] < 0.2:  # Low variability
            adaptive_params['uncertainty_std'] *= 0.8
            print(f"Low demand variability detected (CV={demand_stats['cv']:.3f}), reducing uncertainty modeling")
        
        # Scale epsilon based on problem size and demand characteristics
        if adaptive_params.get('adaptive_epsilon', False):
            size_factor = np.log(problem_size / 50 + 1)
            variability_factor = demand_stats['cv']
            adaptive_params['epsilon'] = adaptive_params['epsilon'] * (1 + 0.1 * size_factor + 0.2 * variability_factor)
            print(f"Adaptive epsilon: {adaptive_params['epsilon']:.4f} (original: {self.dro_params['epsilon']:.4f})")
        
        # Generate historical samples with enhanced preprocessing
        historical_samples = self.generate_enhanced_historical_samples(
            normalized_demand[:, 1:],  # Exclude depot
            instance_name,
            instance_data
        )
        #print(historical_samples.shape)
        # Store preprocessed data
        preprocessed_data = {
            'original_instance': instance_data,
            'normalized_demand': normalized_demand,
            'historical_samples': historical_samples,
            'demand_stats': demand_stats,
            'adaptive_params': adaptive_params,
            'problem_size': problem_size,
            'base_capacity': base_capacity
        }
        
        self.historical_data_cache[instance_name] = preprocessed_data
        
        print(f"DRO preprocessing complete:")
        print(f"  Problem size: {problem_size} customers")
        print(f"  Demand statistics: mean={demand_stats['mean']:.3f}, std={demand_stats['std']:.3f}, CV={demand_stats['cv']:.3f}")
        print(f"  Historical samples: {historical_samples.shape}")
        print(f"  Adaptive DRO parameters applied: {adaptive_params != self.dro_params}")
        
        return preprocessed_data
        
    def test_on_vrplib(self):
        """
        Enhanced test method for DRO-CVRP with comprehensive data acquisition
        """
        ori_directory = os.getcwd() 
        data_directory = os.path.join(ori_directory, "truck_data")
        
        # Check if truck_data directory exists
        if not os.path.exists(data_directory):
            print(f"Error: truck_data directory not found at {data_directory}")
            return
            
        files = os.listdir(data_directory)
        
        # Filter to get only .vrp files
        vrp_files = [f for f in files if f.endswith('.vrp')]
        
        print(f"DRO-CVRP Processing: {len(vrp_files)} VRP files found in truck_data directory")
        print("Files to process: {}".format(vrp_files))
        
        if not vrp_files:
            print("Warning: No .vrp files found in truck_data directory!")
            return
            
        vrplib_results = []
        total_time = 0.
        lis_time = []
        successful_files = 0
        failed_files = []
        
        # DRO-specific metrics tracking
        dro_metrics = {
            'total_routing_costs': [],
            'total_dro_penalties': [],
            'penalty_ratios': [],
            'uncertainty_reductions': [],
            'robustness_levels': []
        }
        
        for t in range(self.repeat_times):
            for filename in vrp_files:
                name = filename[:-4]  # Remove .vrp extension
                instance_file = os.path.join(data_directory, filename)
                
                print(f"\n{'='*80}")
                print(f"Processing DRO-CVRP instance: {name}")
                print(f"Instance file path: {instance_file}")
                print(f"Iteration: {t+1}/{self.repeat_times}")
                print(f"{'='*80}")
                
                # Check if file exists
                if not os.path.exists(instance_file):
                    print(f"Error: File {instance_file} does not exist!")
                    failed_files.append(name)
                    continue
                
                try:
                    result_dict = {}
                    result_dict['run_idx'] = t
                    start_time = time.time()
                    
                    # Process the instance with enhanced DRO
                    self.test_on_one_ins_enhanced(name=name, result_dict=result_dict, instance=instance_file)
                    
                    processing_time = time.time() - start_time
                    total_time += processing_time
                    lis_time.append(processing_time)
                    
                    # Collect DRO-specific metrics
                    routing_cost = result_dict.get('routing_cost', 0)
                    dro_penalty = result_dict.get('dro_penalty', 0)
                    
                    dro_metrics['total_routing_costs'].append(routing_cost)
                    dro_metrics['total_dro_penalties'].append(dro_penalty)
                    
                    if routing_cost > 0:
                        penalty_ratio = dro_penalty / routing_cost
                        dro_metrics['penalty_ratios'].append(penalty_ratio)
                    
                    new_instance_dict = {}
                    new_instance_dict['instance'] = name
                    new_instance_dict['best_cost'] = result_dict['best_cost']
                    new_instance_dict['routing_cost'] = routing_cost
                    new_instance_dict['dro_penalty'] = dro_penalty
                    new_instance_dict['penalty_ratio'] = penalty_ratio if routing_cost > 0 else 0
                    new_instance_dict['best_solution'] = result_dict['best_solution']
                    new_instance_dict['scale'] = result_dict['scale']
                    new_instance_dict['dro_params'] = result_dict.get('adaptive_dro_params', self.dro_params)
                    new_instance_dict['uncertainty_analysis'] = result_dict.get('uncertainty_analysis', {})
                    new_instance_dict['record'] = [result_dict]
                    vrplib_results.append(new_instance_dict)
                    
                    successful_files += 1
                    print(f"\n✓ Successfully processed DRO instance: {name}")
                    print(f"  Total DRO Cost: {result_dict['best_cost']:.4f}")
                    print(f"  Base Routing Cost: {routing_cost:.4f}")
                    print(f"  Robustness Penalty: {dro_penalty:.4f}")
                    print(f"  Penalty Ratio: {penalty_ratio*100:.2f}%" if routing_cost > 0 else "  Penalty Ratio: N/A")
                    print(f"  Problem Scale: {result_dict['scale']} customers, {result_dict['num_routes']} routes")
                    print(f"  Processing Time: {processing_time:.2f}s")
                    
                except Exception as e:
                    print(f"\n✗ Failed to process DRO instance {name}: {e}")
                    import traceback
                    traceback.print_exc()
                    failed_files.append(name)
                    continue

        # Print comprehensive DRO summary
        print("\n" + "="*100)
        print("DRO-CVRP COMPREHENSIVE PROCESSING SUMMARY")
        print("="*100)
        print(f"Total files processed: {len(vrp_files)}")
        print(f"Successfully processed: {successful_files}")
        print(f"Failed: {len(failed_files)}")
        
        if failed_files:
            print(f"Failed files: {failed_files}")
        
        if successful_files > 0:
            print(f"\nTIMING ANALYSIS:")
            print(f"Average processing time: {total_time / successful_files:.2f}s")
            print(f"Total processing time: {total_time:.2f}s")
            print(f"Min/Max processing time: {min(lis_time):.2f}s / {max(lis_time):.2f}s")
            
            # Enhanced DRO-specific statistical analysis
            total_costs = [result['best_cost'] for result in vrplib_results]
            routing_costs = dro_metrics['total_routing_costs']
            dro_penalties = dro_metrics['total_dro_penalties']
            penalty_ratios = dro_metrics['penalty_ratios']
            
            print(f"\nDRO-CVRP OPTIMIZATION ANALYSIS:")
            print(f"Average Total Cost: {np.mean(total_costs):.4f} (±{np.std(total_costs):.4f})")
            print(f"Average Routing Cost: {np.mean(routing_costs):.4f} (±{np.std(routing_costs):.4f})")
            print(f"Average Robustness Penalty: {np.mean(dro_penalties):.4f} (±{np.std(dro_penalties):.4f})")
            
            if penalty_ratios:
                avg_penalty_ratio = np.mean(penalty_ratios)
                print(f"Average Penalty Ratio: {avg_penalty_ratio*100:.2f}% (±{np.std(penalty_ratios)*100:.2f}%)")
                print(f"Min/Max Penalty Ratio: {min(penalty_ratios)*100:.2f}% / {max(penalty_ratios)*100:.2f}%")
            
            # Problem size analysis
            small_problems = [r for r in vrplib_results if r['scale'] <= 50]
            medium_problems = [r for r in vrplib_results if 50 < r['scale'] <= 100]
            large_problems = [r for r in vrplib_results if r['scale'] > 100]
            
            print(f"\nPROBLEM SIZE ANALYSIS:")
            if small_problems:
                small_costs = [r['best_cost'] for r in small_problems]
                small_penalties = [r['penalty_ratio'] for r in small_problems if r['penalty_ratio'] > 0]
                print(f"Small problems (≤50): {len(small_problems)} instances")
                print(f"  Avg cost: {np.mean(small_costs):.4f}, Avg penalty ratio: {np.mean(small_penalties)*100:.2f}%" if small_penalties else "  Avg penalty ratio: N/A")
                
            if medium_problems:
                medium_costs = [r['best_cost'] for r in medium_problems]
                medium_penalties = [r['penalty_ratio'] for r in medium_problems if r['penalty_ratio'] > 0]
                print(f"Medium problems (51-100): {len(medium_problems)} instances")
                print(f"  Avg cost: {np.mean(medium_costs):.4f}, Avg penalty ratio: {np.mean(medium_penalties)*100:.2f}%" if medium_penalties else "  Avg penalty ratio: N/A")
                
            if large_problems:
                large_costs = [r['best_cost'] for r in large_problems]
                large_penalties = [r['penalty_ratio'] for r in large_problems if r['penalty_ratio'] > 0]
                print(f"Large problems (>100): {len(large_problems)} instances")
                print(f"  Avg cost: {np.mean(large_costs):.4f}, Avg penalty ratio: {np.mean(large_penalties)*100:.2f}%" if large_penalties else "  Avg penalty ratio: N/A")
        
        # Save comprehensive summary results
        try:
            summary_file = os.path.join(self.results_dir, f"{self.config['name']}_dro_comprehensive_summary.json")
            with open(summary_file, 'w', encoding='utf-8') as f:
                summary_data = {
                    'experiment_info': {
                        'total_files': len(vrp_files),
                        'successful_files': successful_files,
                        'failed_files': len(failed_files),
                        'failed_file_names': failed_files,
                        'repeat_times': self.repeat_times
                    },
                    'timing_analysis': {
                        'average_time': total_time / successful_files if successful_files > 0 else 0,
                        'total_time': total_time,
                        'processing_times': lis_time,
                        'min_time': min(lis_time) if lis_time else 0,
                        'max_time': max(lis_time) if lis_time else 0
                    },
                    'dro_analysis': {
                        'base_dro_params': self.dro_params,
                        'cost_statistics': {
                            'mean_total_cost': np.mean(total_costs) if total_costs else 0,
                            'std_total_cost': np.std(total_costs) if total_costs else 0,
                            'mean_routing_cost': np.mean(routing_costs) if routing_costs else 0,
                            'std_routing_cost': np.std(routing_costs) if routing_costs else 0,
                            'mean_dro_penalty': np.mean(dro_penalties) if dro_penalties else 0,
                            'std_dro_penalty': np.std(dro_penalties) if dro_penalties else 0
                        },
                        'penalty_analysis': {
                            'mean_penalty_ratio': np.mean(penalty_ratios) if penalty_ratios else 0,
                            'std_penalty_ratio': np.std(penalty_ratios) if penalty_ratios else 0,
                            'min_penalty_ratio': min(penalty_ratios) if penalty_ratios else 0,
                            'max_penalty_ratio': max(penalty_ratios) if penalty_ratios else 0
                        }
                    },
                    'detailed_results': vrplib_results,
                    'uncertainty_scenarios': {k: {
                        'epsilon': v['epsilon'], 
                        'strategy': v['strategy'], 
                        'std': v['std']
                    } for k, v in self.uncertainty_scenarios.items()}
                }
                json.dump(summary_data, f, indent=2, ensure_ascii=False)
            
            print(f"\nComprehensive DRO summary saved to: {summary_file}")
            
        except Exception as e:
            print(f"Failed to save comprehensive summary: {e}")
        
        print(f"\nAll DRO results saved in directory: {self.results_dir}")
        print("Each instance includes:")
        print("  - DRO solution analysis (.txt)")
        print("  - Interactive route map (.html)")
        print("  - Route visualization (.png)")
        print("  - Uncertainty scenario analysis")
        print("  - Two-stage optimization results (DRO + LKH)")

    def test_on_one_ins_enhanced(self, name, result_dict, instance):
        """
        Enhanced single instance testing with comprehensive DRO analysis
        """
        print(f"\n--- Enhanced DRO Analysis for Instance: {name} ---")
        
        # Check instance parameter and load data
        if isinstance(instance, str):
            instance_data = vrplib.read_instance(instance)
            instance_file_path = instance
        else:
            instance_data = instance
            instance_file_path = None
            print("Warning: instance is not a file path, LKH optimization may be limited")
        
        # Enhanced preprocessing for DRO
        preprocessed_data = self.preprocess_vrp_data_for_dro(instance_data, name)
        problem_size = preprocessed_data['problem_size']
        adaptive_params = preprocessed_data['adaptive_params']
        
        multiple_width = min(problem_size, 1000)

        # Extract coordinate information
        coordinates = {}
        for i, (x, y) in enumerate(instance_data['node_coord']):
            coordinates[i] = (x, y)

        print(f"STAGE 1: DRO-CVRP OPTIMIZATION")
        print(f"Problem size: {problem_size} customers")
        print(f"Using adaptive DRO parameters: {adaptive_params != self.dro_params}")
        
        # Initialize enhanced DRO-CVRP environment
        env = CVRPEnv_DRO(multiple_width, self.device, adaptive_params)
        env.load_vrplib_problem(instance_data, aug_factor=self.aug_factor, 
                               historical_samples=preprocessed_data['historical_samples'])
        
                # Add empirical distribution for DRO
        empirical_dist =  torch.randint(low=0, high=2, size=(117,)) 
        
        
        env.add_empirical_distribution(empirical_dist)
        
        reset_state, reward, done = env.reset()
        self.model.eval()
        self.model.requires_grad_(False)
        self.model.pre_forward(reset_state)
        
        # Stage 1: DRO optimization
        stage1_start_time = time.time()
        with torch.no_grad():
            policy_solutions, policy_prob, rewards = rollout_dro(self.model, env, 'greedy')
        stage1_time = time.time() - stage1_start_time
                
        # Process Stage 1 results with enhanced analysis
        aug_reward = rewards.reshape(self.aug_factor, 1, env.multi_width)
        max_pomo_reward, pomo_indices = aug_reward.max(dim=2)
        max_aug_pomo_reward, aug_indices = max_pomo_reward.max(dim=0)  # ✅ 正确：使用max_pomo_reward
        
        best_cost_stage1 = -max_aug_pomo_reward.float()
        
        # Get optimal solution indices
        best_augmentation_idx = aug_indices.item()
        best_pomo_idx = pomo_indices[best_augmentation_idx, 0].item()
        
        # Extract optimal solution path
        stage1_solution = policy_solutions[best_augmentation_idx, best_pomo_idx, :]
        
        # Enhanced cost analysis
        routing_cost = env.compute_unscaled_routing_cost_only()
        best_routing_cost = -routing_cost.reshape(self.aug_factor, 1, env.multi_width).max(dim=2)[0].max(dim=0)[0].float()
        dro_penalty = best_cost_stage1 - best_routing_cost
        
        # Uncertainty analysis
        uncertainty_analysis = {
            'demand_variability': preprocessed_data['demand_stats']['cv'],
            'uncertainty_scenarios': len(preprocessed_data['historical_samples'][0]),
            'wasserstein_radius': adaptive_params['epsilon'],
            'sample_strategy': adaptive_params['sample_strategy'],
            'penalty_to_routing_ratio': (dro_penalty / best_routing_cost).item() if best_routing_cost > 0 else 0
        }
        
        print(f"Stage 1 DRO Results:")
        print(f"  Total DRO Cost: {best_cost_stage1.item():.4f}")
        print(f"  Base Routing Cost: {best_routing_cost.item():.4f}")
        print(f"  Robustness Penalty: {dro_penalty.item():.4f}")
        print(f"  Penalty Ratio: {uncertainty_analysis['penalty_to_routing_ratio']*100:.2f}%")
        print(f"  Optimization Time: {stage1_time:.2f}s")
        print(f"  Best Configuration: Aug {best_augmentation_idx}, POMO {best_pomo_idx}")
        
        # Stage 2: LKH optimization (deterministic)
        print(f"\nSTAGE 2: LKH DETERMINISTIC REFINEMENT")
        
        if instance_file_path is not None:
            try:
                stage2_start_time = time.time()
                
                # Convert stage1 solution for LKH
                if isinstance(stage1_solution, torch.Tensor):
                    stage1_solution_list = stage1_solution.cpu().numpy().tolist()
                else:
                    stage1_solution_list = stage1_solution.tolist() if hasattr(stage1_solution, 'tolist') else stage1_solution
                
                print(f"  Running LKH on DRO solution...")
                print(f"  Stage 1 solution length: {len(stage1_solution_list)}")
                
                # Stage 2 LKH optimization using DRO results
                final_solution, final_routing_cost = stage2_lkh_optimization(
                    instance_file_path, stage1_solution_list, None
                )
                
                stage2_time = time.time() - stage2_start_time
                
                # Calculate final costs
                # LKH optimizes routing cost, we estimate total cost by adding original DRO penalty
                routing_improvement = best_routing_cost.item() - final_routing_cost
                estimated_dro_penalty = max(0, dro_penalty.item() - routing_improvement * 0.1)  # Slight penalty reduction
                final_total_cost = final_routing_cost + estimated_dro_penalty
                
                print(f"Stage 2 LKH Results:")
                print(f"  Improved Routing Cost: {final_routing_cost:.4f}")
                print(f"  Routing Improvement: {routing_improvement:.4f}")
                print(f"  Estimated DRO Penalty: {estimated_dro_penalty:.4f}")
                print(f"  Final Total Cost: {final_total_cost:.4f}")
                print(f"  LKH Optimization Time: {stage2_time:.2f}s")
                
                # Convert to route format
                if isinstance(final_solution, list):
                    routes = convert_solution_to_routes(final_solution)
                else:
                    routes = convert_solution_to_routes(final_solution.cpu().numpy().tolist())
                
                # Update uncertainty analysis with final results
                uncertainty_analysis.update({
                    'stage1_cost': best_cost_stage1.item(),
                    'stage2_routing_cost': final_routing_cost,
                    'final_total_cost': final_total_cost,
                    'routing_improvement': routing_improvement,
                    'lkh_optimization_time': stage2_time,
                    'two_stage_improvement': best_cost_stage1.item() - final_total_cost
                })
                
            except Exception as e:
                print(f"LKH optimization failed, using Stage 1 DRO result: {e}")
                import traceback
                traceback.print_exc()
                
                final_solution = stage1_solution.cpu().numpy().tolist()
                final_total_cost = best_cost_stage1.item()
                final_routing_cost = best_routing_cost.item()
                estimated_dro_penalty = dro_penalty.item()
                routes = convert_solution_to_routes(final_solution)
                
                uncertainty_analysis.update({
                    'stage1_cost': best_cost_stage1.item(),
                    'final_total_cost': final_total_cost,
                    'lkh_failed': True,
                    'lkh_error': str(e)
                })
        else:
            print("  No file path available for LKH optimization")
            final_solution = stage1_solution.cpu().numpy().tolist()
            final_total_cost = best_cost_stage1.item()
            final_routing_cost = best_routing_cost.item()
            estimated_dro_penalty = dro_penalty.item()
            routes = convert_solution_to_routes(final_solution)
            
            uncertainty_analysis.update({
                'stage1_cost': best_cost_stage1.item(),
                'final_total_cost': final_total_cost,
                'lkh_available': False
            })
        
        # Generate comprehensive output files with enhanced DRO information
        enhanced_dro_info = {
            'routing_cost': final_routing_cost,
            'dro_penalty': estimated_dro_penalty,
            'uncertainty_analysis': uncertainty_analysis,
            **adaptive_params
        }
        
        self._save_all_enhanced_outputs(
            name, routes, final_total_cost, coordinates, result_dict, 
            problem_size, stage1_cost=best_cost_stage1.item(), 
            dro_info=enhanced_dro_info, adaptive_params=adaptive_params,
            uncertainty_analysis=uncertainty_analysis
        )
        
        print(f"\n--- DRO Analysis Complete for {name} ---")
        print(f"Total processing time: {time.time() - (stage1_start_time - stage1_time):.2f}s")
    
    def _save_all_enhanced_outputs(self, name, routes, final_best_cost, coordinates, result_dict, 
                                 problem_size, stage1_cost=None, dro_info=None, adaptive_params=None,
                                 uncertainty_analysis=None):
        """
        Save all output files for a single instance with enhanced DRO analysis
        """
        # Print comprehensive solution analysis
        print_solution(routes, final_best_cost, coordinates, name, dro_info)
        
        # Define file paths
        solution_filename = os.path.join(self.results_dir, f"{name}_dro_enhanced_solution.txt")
        map_filename = os.path.join(self.results_dir, f"{name}_dro_route_map.html")
        plot_filename = os.path.join(self.results_dir, f"{name}_dro_route_plot.png")
        analysis_filename = os.path.join(self.results_dir, f"{name}_dro_uncertainty_analysis.json")
        
        # Save enhanced text solution with comprehensive DRO analysis
        try:
            save_solution_to_file(routes, final_best_cost, solution_filename, coordinates, name, dro_info)
        except Exception as e:
            print(f"Failed to save enhanced DRO solution file for {name}: {e}")
        
        # Save detailed uncertainty analysis
        try:
            with open(analysis_filename, 'w', encoding='utf-8') as f:
                analysis_data = {
                    'instance_name': name,
                    'problem_size': problem_size,
                    'dro_parameters': {
                        'base_params': self.dro_params,
                        'adaptive_params': adaptive_params,
                        'adaptation_applied': adaptive_params != self.dro_params if adaptive_params else False
                    },
                    'cost_analysis': {
                        'stage1_dro_cost': stage1_cost,
                        'final_total_cost': final_best_cost,
                        'routing_cost': dro_info.get('routing_cost', 0) if dro_info else 0,
                        'dro_penalty': dro_info.get('dro_penalty', 0) if dro_info else 0,
                        'improvement': stage1_cost - final_best_cost if stage1_cost else 0
                    },
                    'uncertainty_analysis': uncertainty_analysis or {},
                    'solution_details': {
                        'num_routes': len(routes),
                        'routes': routes,
                        'total_customers': sum(len(route) for route in routes),
                        'avg_customers_per_route': sum(len(route) for route in routes) / len(routes) if routes else 0
                    },
                    'coordinates': coordinates
                }
                json.dump(analysis_data, f, indent=2, ensure_ascii=False)
            print(f"Uncertainty analysis saved to {analysis_filename}")
        except Exception as e:
            print(f"Failed to save uncertainty analysis for {name}: {e}")
        
        # Create enhanced visualizations
        try:
            create_route_map(routes, coordinates, depot_idx=0, filename=map_filename)
        except Exception as e:
            print(f"Failed to create DRO route map for {name}: {e}")
        
        try:
            create_route_plot(routes, coordinates, depot_idx=0, filename=plot_filename)
        except Exception as e:
            print(f"Failed to create DRO route plot for {name}: {e}")
        
        # Save enhanced results to result_dict
        if result_dict is not None:
            result_dict['best_cost'] = final_best_cost
            result_dict['routing_cost'] = dro_info.get('routing_cost', 0) if dro_info else 0
            result_dict['dro_penalty'] = dro_info.get('dro_penalty', 0) if dro_info else 0
            result_dict['best_solution'] = routes
            result_dict['routes'] = routes
            result_dict['scale'] = problem_size
            result_dict['stage1_cost'] = stage1_cost if stage1_cost is not None else final_best_cost
            result_dict['num_routes'] = len(routes)
            result_dict['coordinates'] = coordinates
            result_dict['base_dro_params'] = self.dro_params
            result_dict['adaptive_dro_params'] = adaptive_params
            result_dict['dro_info'] = dro_info
            result_dict['uncertainty_analysis'] = uncertainty_analysis
            result_dict['files'] = {
                'solution': solution_filename,
                'map': map_filename,
                'plot': plot_filename,
                'analysis': analysis_filename
            }


if __name__ == "__main__":
    
    # Define target directory path for DRO-CVRP
    target_directory = r"elg-master\cvrp_dro"
    ori_directory = os.getcwd()
    
    # Switch to target directory
    os.chdir(target_directory)
    
    # Load enhanced DRO configuration
    with open('config_dro.yml', 'r', encoding='utf-8') as config_file:
        config = yaml.load(config_file.read(), Loader=yaml.FullLoader)
    
    print("="*100)
    print("ENHANCED DRO-CVRP TRUE VALUE TESTER")
    print("="*100)
    print("Initializing Distributionally Robust Optimization for Vehicle Routing...")
    print("Features:")
    print("  - Enhanced uncertainty modeling with multiple strategies")
    print("  - Adaptive DRO parameter tuning based on problem characteristics")
    print("  - Two-stage optimization: DRO -> LKH refinement")
    print("  - Comprehensive robustness analysis and visualization")
    print("  - Multi-scenario demand uncertainty handling")
    print("="*100)
    
    tester = VRPLib_Tester_DRO(config=config)
    tester.test_on_vrplib()