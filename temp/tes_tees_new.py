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
from typing import List, Tuple, Dict
from KLH import convert_solution_to_sequence, stage2_lkh_optimization,  parse_stage1_solution  
import warnings
import folium
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import pandas as pd

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
    Create route map visualization
    Args:
        routes: List of routes
        coordinates: Coordinate dictionary {node_id: (lon, lat)}
        depot_idx: Depot node index
        filename: HTML filename to save
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
    
    # Create map with English labels only
    m = folium.Map(
        location=[center_lat, center_lon], 
        zoom_start=12,
        tiles=None  # Don't use default tiles
    )
    
    # Add multiple English-only map layers
    try:
        # Option 1: CartoDB Positron (clean English labels)
        folium.TileLayer(
            tiles='https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',
            attr='© OpenStreetMap contributors © CARTO',
            name='CartoDB Positron (English)',
            overlay=False,
            control=True,
            subdomains='abcd'
        ).add_to(m)
        
        # Option 2: OpenStreetMap (English)
        folium.TileLayer(
            tiles='https://tile.openstreetmap.org/{z}/{x}/{y}.png',
            attr='© OpenStreetMap contributors',
            name='OpenStreetMap (English)',
            overlay=False,
            control=True
        ).add_to(m)
        
        # Option 3: Stamen Terrain (English labels)
        folium.TileLayer(
            tiles='https://stamen-tiles-{s}.a.ssl.fastly.net/terrain/{z}/{x}/{y}{r}.png',
            attr='Map tiles by Stamen Design, CC BY 3.0 — Map data © OpenStreetMap contributors',
            name='Stamen Terrain (English)',
            overlay=False,
            control=True,
            subdomains='abcd'
        ).add_to(m)
        
        # Option 4: ESRI World Street Map (English)
        folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}',
            attr='Tiles © Esri',
            name='ESRI World Street Map (English)',
            overlay=False,
            control=True
        ).add_to(m)
        
    except Exception as e:
        print(f"Failed to add custom tile layers, using default: {e}")
        # Fallback to default OpenStreetMap
        folium.TileLayer(
            tiles='OpenStreetMap',
            name='OpenStreetMap',
            overlay=False,
            control=True
        ).add_to(m)
    
    # Define color list for different routes
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred',
              'beige', 'darkblue', 'darkgreen', 'cadetblue', 'darkpurple', 'white',
              'pink', 'lightblue', 'lightgreen', 'gray', 'black', 'lightgray']
    
    # Add depot marker
    depot_coord = converted_coords[depot_idx]
    folium.Marker(
        depot_coord, 
        popup=f"Depot (Node {depot_idx})",
        tooltip=f"Depot Location",
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
        if len(route_coords) > 1:
            folium.PolyLine(
                route_coords,
                color=color,
                weight=3,
                opacity=0.8,
                popup=f"Route #{i+1}",
                tooltip=f"Route #{i+1} - {len(route)} customers"
            ).add_to(m)
        
        # Add customer markers
        for j, node_id in enumerate(route):
            if node_id in converted_coords:
                folium.CircleMarker(
                    converted_coords[node_id],
                    radius=6,
                    popup=f"Route #{i+1}, Stop {j+1}, Node {node_id}",
                    tooltip=f"Customer {node_id}",
                    color=color,
                    fillColor=color,
                    fillOpacity=0.7
                ).add_to(m)
    
    # Save map
    m.save(filename)
    print(f"Route map saved to {filename}")
    
    return m

def create_route_plot(routes, coordinates, depot_idx=0, filename="route_plot.png"):
    """
    Create route matplotlib plot
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
        if len(route_lons) > 1:
            plt.plot(route_lons, route_lats, color=color, linewidth=2, 
                    label=f'Route #{i+1}', marker='o', markersize=4)
    
    # Mark depot
    if depot_idx in converted_coords:
        depot_lon, depot_lat = converted_coords[depot_idx]
        plt.plot(depot_lon, depot_lat, 'ks', markersize=12, label='Depot')
    
    plt.xlabel('Longitude (°)')
    plt.ylabel('Latitude (°)')
    plt.title('Vehicle Routing Problem - Route Visualization')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Route plot saved to {filename}")

def convert_solution_to_routes(solution_sequence, depot_idx=0):
    """
    Convert solution sequence to route format
    Args:
        solution_sequence: Solution sequence containing node visit order
        depot_idx: Depot node index, default is 0
    Returns:
        routes: List of routes, each route is a list of nodes
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

def save_solution_to_file(routes, cost, filename, coordinates=None):
    """
    Save route solution to file with coordinate information
    Args:
        routes: List of routes
        cost: Total cost
        filename: Save filename
        coordinates: Coordinate dictionary
    """
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("=== Vehicle Routing Problem Solution ===\n\n")
        
        for i, route in enumerate(routes, 1):
            route_str = ' '.join(map(str, route))
            f.write(f"Route #{i}: {route_str}\n")
            
            # If coordinate information exists, save coordinates too
            if coordinates:
                f.write(f"  Coordinates: ")
                for node_id in route:
                    if node_id in coordinates:
                        x, y = coordinates[node_id]
                        lon = convert_coordinates(x)
                        lat = convert_coordinates(y)
                        f.write(f"Node{node_id}({lon:.5f},{lat:.5f}) ")
                f.write("\n")
        
        f.write(f"\nTotal Cost: {cost:.0f}\n")
        f.write(f"Number of Routes: {len(routes)}\n")
    
    print(f"Solution saved to {filename}")

def print_solution(routes, cost, coordinates=None):
    """
    Print route solution with coordinate information
    """
    print("\n=== Route Planning Solution ===")
    for i, route in enumerate(routes, 1):
        route_str = ' '.join(map(str, route))
        print(f"Route #{i}: {route_str}")
        
        # If coordinate information exists, print coordinates too
        if coordinates:
            print(f"  Coordinates: ", end="")
            for node_id in route:
                if node_id in coordinates:
                    x, y = coordinates[node_id]
                    lon = convert_coordinates(x)
                    lat = convert_coordinates(y)
                    print(f"Node{node_id}({lon:.5f},{lat:.5f}) ", end="")
            print()
    
    print(f"Total Cost: {cost:.0f}")
    print(f"Number of Routes: {len(routes)}")

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
        self.vrplib_path = r'test' 
        self.repeat_times = 1
        self.aug_factor = config['params']['aug_factor']
        self.vrplib_results = None
        
    def test_on_vrplib(self):
               
        ori_directory = os.getcwd() 
        data_directory = os.path.join(ori_directory, "truck_data")
        files = os.listdir(data_directory)
        
        # Filter to get only .vrp files
        vrp_files = [f for f in files if f.endswith('.vrp')]
        
        print("Processing {} VRP files found in truck_data directory".format(len(vrp_files)))
        print("Files to process: {}".format(vrp_files))
        
        if not vrp_files:
            print("Warning: No .vrp files found in truck_data directory!")
            return
            
        vrplib_results = []
        total_time = 0.
        lis_time = []
        
        for t in range(self.repeat_times):
            for filename in vrp_files:
                name = filename[:-4]  # Remove .vrp extension
                instance_file = os.path.join(data_directory, filename)  # Use correct path
                
                print(f"Processing instance: {name}")
                print(f"Instance file path: {instance_file}")
                
                # Check if file exists
                if not os.path.exists(instance_file):
                    print(f"Error: File {instance_file} does not exist!")
                    continue
                
                result_dict = {}
                result_dict['run_idx'] = t
                start_time = time.time()
                self.test_on_one_ins(name=name, result_dict=result_dict, instance=instance_file)
                total_time += time.time() - start_time
                lis_time.append(time.time() - start_time)
                
                new_instance_dict = {}
                new_instance_dict['instance'] = name
                new_instance_dict['best_cost'] = result_dict['best_cost']
                new_instance_dict['best_solution'] = result_dict['best_solution']
                new_instance_dict['scale'] = result_dict['scale']
                new_instance_dict['record'] = [result_dict]
                vrplib_results.append(new_instance_dict)
                
                print("Instance Name {}: Best Cost {:.4f}".format(name, result_dict['best_cost']))
                print("Scale: {} customers, {} routes".format(result_dict['scale'], result_dict['num_routes']))
                print('=========================================')

        print("Average time: {:.2f}s".format(total_time / len(vrp_files)))
        print("Total time: {:.2f}s".format(total_time))
        print("Processing times: ", lis_time)
        
        # Save detailed results to JSON
        os.makedirs('test_results', exist_ok=True)
        with open('test_results/' + self.config['name'] + '_' + 'vrplib_results.json', 'w', encoding='utf-8') as f:
            json.dump(vrplib_results, f, indent=2, ensure_ascii=False)
        
        print("Detailed results saved to test_results/{}_vrplib_results.json".format(self.config['name']))
        print("Individual solution files saved to test_results/[instance_name]_solution.txt")

    def test_on_one_ins(self, name, result_dict, instance):
        # Check instance parameter
        print(f"Instance parameter type: {type(instance)}")
        print(f"Instance parameter value: {instance}")
        
        # If instance is a string path, read the instance
        if isinstance(instance, str):
            instance_data = vrplib.read_instance(instance)
            instance_file_path = instance  # Save file path
        else:
            # If instance is already data, use directly
            instance_data = instance
            instance_file_path = None
            print("Warning: instance is not a file path, LKH optimization may fail")
        
        problem_size = instance_data['node_coord'].shape[0] - 1
        multiple_width = min(problem_size, 1000)

        # Extract coordinate information
        coordinates = {}
        for i, (x, y) in enumerate(instance_data['node_coord']):
            coordinates[i] = (x, y)

        # Initialize CVRP state
        env = CVRPEnv(multiple_width, self.device)
        env.load_vrplib_problem(instance_data, aug_factor=self.aug_factor)
        
        reset_state, reward, done = env.reset()
        self.model.eval()
        self.model.requires_grad_(False)
        self.model.pre_forward(reset_state)
        
        with torch.no_grad():
            policy_solutions, policy_prob, rewards = rollout(self.model, env, 'greedy')
                
        # Return
        aug_reward = rewards.reshape(self.aug_factor, 1, env.multi_width)
        
        # shape: (augmentation, batch, multi)
        max_pomo_reward, pomo_indices = aug_reward.max(dim=2)  # get best results from pomo
        
        # shape: (augmentation, batch)
        max_aug_pomo_reward, aug_indices = max_pomo_reward.max(dim=0)  # get best results from augmentation
        
        # shape: (batch,)
        best_cost_stage1 = -max_aug_pomo_reward.float()  # negative sign to make positive value
        
        # Get optimal solution indices
        best_augmentation_idx = aug_indices.item()
        best_pomo_idx = pomo_indices[best_augmentation_idx, 0].item()
        
        # Extract optimal solution path
        stage1_solution = policy_solutions[best_augmentation_idx, best_pomo_idx, :]
        
        print(f"Stage 1 calculated best cost (path length): {best_cost_stage1.item():.4f}")
        print(f"Corresponding augmentation index for this best cost: {best_augmentation_idx}")
        print(f"Corresponding POMO index for this best cost: {best_pomo_idx}")
        
        # Stage 2 LKH optimization
        if instance_file_path is not None:
            vrp_file = instance_file_path  # Use file path
            print(f"VRP file path: {vrp_file}")
            print(f"VRP file path type: {type(vrp_file)}")
        else:
            print("Cannot perform LKH optimization: no file path")
            final_solution = stage1_solution.cpu().numpy().tolist()
            final_best_cost = best_cost_stage1.item()
            routes = convert_solution_to_routes(final_solution)
            print("\n=== Final Solution (Stage 1 only) ===")
            print_solution(routes, final_best_cost, coordinates)
            
            # Save to file
            solution_filename = f"test_results/{name}_solution.txt"
            os.makedirs('test_results', exist_ok=True)
            save_solution_to_file(routes, final_best_cost, solution_filename, coordinates)
            
            # Create map visualization
            map_filename = f"test_results/{name}_route_map.html"
            plot_filename = f"test_results/{name}_route_plot.png"
            create_route_map(routes, coordinates, depot_idx=0, filename=map_filename)
            create_route_plot(routes, coordinates, depot_idx=0, filename=plot_filename)
            
            # Save results and return
            if result_dict is not None:
                result_dict['best_cost'] = final_best_cost
                result_dict['best_solution'] = final_solution
                result_dict['routes'] = routes
                result_dict['scale'] = problem_size
                result_dict['stage1_cost'] = best_cost_stage1.item()
                result_dict['num_routes'] = len(routes)
                result_dict['coordinates'] = coordinates
            return
        
        print(f"Stage1 solution type: {type(stage1_solution)}")
        print(f"Stage1 solution shape: {stage1_solution.shape if hasattr(stage1_solution, 'shape') else 'No shape'}")
        
        try:
            # Ensure stage1_solution is in correct format
            if isinstance(stage1_solution, torch.Tensor):
                stage1_solution_list = stage1_solution.cpu().numpy().tolist()
            else:
                stage1_solution_list = stage1_solution.tolist() if hasattr(stage1_solution, 'tolist') else stage1_solution
            
            print(f"Converted Stage1 solution type: {type(stage1_solution_list)}")
            print(f"Stage1 solution first 10 elements: {stage1_solution_list[:10]}")
            
            final_solution, final_best_cost = stage2_lkh_optimization(vrp_file, stage1_solution_list, None)  # Pass None as optimal
            print(f"LKH optimized final solution: {final_solution}")
            print(f"LKH optimized final cost: {final_best_cost:.4f}")
            
            # Convert to route format
            if isinstance(final_solution, list):
                routes = convert_solution_to_routes(final_solution)
            else:
                routes = convert_solution_to_routes(final_solution.cpu().numpy().tolist())
            
        except Exception as e:
            print(f"LKH optimization failed, using Stage 1 result: {e}")
            print(f"Error type: {type(e)}")
            import traceback
            traceback.print_exc()
            
            final_solution = stage1_solution.cpu().numpy().tolist()
            final_best_cost = best_cost_stage1.item()
            routes = convert_solution_to_routes(final_solution)
        
        # Print and save solution
        print("\n=== Final Solution ===")
        print_solution(routes, final_best_cost, coordinates)
        
        # Save to file
        solution_filename = f"test_results/{name}_solution.txt"
        os.makedirs('test_results', exist_ok=True)
        save_solution_to_file(routes, final_best_cost, solution_filename, coordinates)
        
        # Create map visualization
        map_filename = f"test_results/{name}_route_map.html"
        plot_filename = f"test_results/{name}_route_plot.png"
        
        try:
            create_route_map(routes, coordinates, depot_idx=0, filename=map_filename)
            create_route_plot(routes, coordinates, depot_idx=0, filename=plot_filename)
            print(f"Map visualizations created: {map_filename}, {plot_filename}")
        except Exception as e:
            print(f"Failed to create visualizations: {e}")
        
        # Save results
        if result_dict is not None:
            result_dict['best_cost'] = final_best_cost
            result_dict['best_solution'] = final_solution if isinstance(final_solution, list) else final_solution.tolist()
            result_dict['routes'] = routes
            result_dict['scale'] = problem_size
            result_dict['stage1_cost'] = best_cost_stage1.item()
            result_dict['num_routes'] = len(routes)
            result_dict['coordinates'] = coordinates


if __name__ == "__main__":
    
    # Define target directory path
    target_directory = r"elg-master\cvrp"
    ori_directory = os.getcwd()
    # Use os.chdir() to switch to target directory
    os.chdir(target_directory)
    
    with open('config.yml', 'r', encoding='utf-8') as config_file:
        config = yaml.load(config_file.read(), Loader=yaml.FullLoader)
    tester = VRPLib_Tester(config=config)
    tester.test_on_vrplib()