import torch
import numpy as np


def rollout_dro(model, env, eval_type):
    """
    Modified rollout function for DRO-CVRP
    """
    env.reset()
    state = env.pre_step()
    done = False
    solutions = []
    probs = []

    while not done:
        # Get current features including visited nodes information
        cur_dist, cur_theta, xy, norm_demand, visited_nodes = env.get_cur_feature()
        
        
        
            # 处理 state 元组问题
        if isinstance(state, tuple):
            # 通常第一个元素是状态对象
            actual_state = state[0]
        else:
            actual_state = state
        
        selected, prob = model.one_step_rollout(actual_state, cur_dist, cur_theta, xy, 
                                                  norm_demand, visited_nodes, eval_type)
        
        state, reward, done = env.step(selected)
        
        solutions.append(selected)
        if prob is not None:
            probs.append(prob)
    
    # Combine solutions
    if len(solutions) > 0:
        solutions = torch.stack(solutions, dim=2)  # shape: (batch, multi, steps)
    else:
        solutions = torch.empty(0)
    
    if len(probs) > 0:
        probs = torch.stack(probs, dim=2)  # shape: (batch, multi, steps)
    else:
        probs = torch.empty(0)
    
    return solutions, probs, reward


def augment_xy_data_by_8_fold(xy_data):
    """
    Augment data by 8-fold using rotations and reflections
    """
    # xy_data.shape: (batch, n, 2)
    
    x = xy_data[:, :, 0]  # shape: (batch, n)
    y = xy_data[:, :, 1]  # shape: (batch, n)
    
    # Original
    dat1 = xy_data
    
    # Rotate 90 degrees
    dat2 = torch.stack((-y, x), dim=2)
    
    # Rotate 180 degrees
    dat3 = torch.stack((-x, -y), dim=2)
    
    # Rotate 270 degrees
    dat4 = torch.stack((y, -x), dim=2)
    
    # Reflect over x-axis
    dat5 = torch.stack((x, -y), dim=2)
    
    # Reflect over y-axis
    dat6 = torch.stack((-x, y), dim=2)
    
    # Reflect over y=x line
    dat7 = torch.stack((y, x), dim=2)
    
    # Reflect over y=-x line
    dat8 = torch.stack((-y, -x), dim=2)
    
    # Concatenate all augmentations
    aug_xy_data = torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=0)
    
    return aug_xy_data


def check_feasible(solutions, demands, capacity=1.0):
    """
    Check if solutions are feasible for CVRP
    """
    batch_size, multi_width, tour_length = solutions.shape
    
    feasible = torch.ones((batch_size, multi_width), dtype=torch.bool, device=solutions.device)
    
    for b in range(batch_size):
        for m in range(multi_width):
            current_load = 0.0
            tour = solutions[b, m, :]
            
            for i in range(tour_length):
                node = tour[i].item()
                
                if node == 0:  # Depot
                    current_load = 0.0  # Reset load at depot
                else:
                    # Add demand of customer node
                    if node - 1 < len(demands[b]):
                        current_load += demands[b][node - 1].item()
                        
                        # Check capacity constraint
                        if current_load > capacity + 1e-6:  # Small tolerance for numerical errors
                            feasible[b, m] = False
                            break
    
    return feasible


def compute_dro_objective_components(visited_nodes, historical_samples, dro_params):
    """
    Compute the components of the DRO objective function
    
    Args:
        visited_nodes: (batch, multi, problem) - binary indicators of visited nodes
        historical_samples: (batch, N, problem) - historical demand samples
        dro_params: dictionary with DRO parameters
    
    Returns:
        lambda_term, beta_terms, unvisited_penalties
    """
    epsilon = dro_params['epsilon']
    M = dro_params['M']
    N = dro_params['N']
    
    batch_size, multi_width, problem_size = visited_nodes.shape
    
    # Initialize dual variables
    lambda_dual = torch.zeros((batch_size, multi_width), device=visited_nodes.device)
    beta_vars = torch.zeros((batch_size, multi_width, N), device=visited_nodes.device)
    
    unvisited_penalties = []
    
    for s in range(N):
        # For sample s, compute penalty for unvisited nodes
        sample_demands = historical_samples[:, s, :].unsqueeze(1)  # shape: (batch, 1, problem)
        sample_demands = sample_demands.expand(-1, multi_width, -1)  # shape: (batch, multi, problem)
        
        # Penalty for unvisited nodes: M * demand * (1 - visited)
        unvisited_mask = 1.0 - visited_nodes  # shape: (batch, multi, problem)
        sample_penalty = (sample_demands * unvisited_mask * M).sum(dim=2)  # shape: (batch, multi)
        
        unvisited_penalties.append(sample_penalty)
        
        # Simple dual variable update (in practice, you might want more sophisticated optimization)
        beta_vars[:, :, s] = torch.clamp(sample_penalty, min=0.0)
    
    # Compute objective components
    lambda_term = epsilon * lambda_dual
    beta_term = beta_vars.mean(dim=2)  # Average over samples
    
    return lambda_term, beta_term, torch.stack(unvisited_penalties, dim=2)


def solve_dual_problem(visited_nodes, historical_samples, dro_params, max_iter=100, lr=0.01):
    """
    Solve the dual problem for DRO optimization
    
    This is a simplified version - in practice you might want to use more sophisticated
    optimization methods like interior point methods or specialized solvers.
    """
    epsilon = dro_params['epsilon']
    M = dro_params['M']
    N = dro_params['N']
    
    batch_size, multi_width, problem_size = visited_nodes.shape
    device = visited_nodes.device
    
    # Initialize dual variables
    lambda_dual = torch.zeros((batch_size, multi_width), device=device, requires_grad=True)
    beta_vars = torch.zeros((batch_size, multi_width, N), device=device, requires_grad=True)
    
    optimizer = torch.optim.Adam([lambda_dual, beta_vars], lr=lr)
    
    for iteration in range(max_iter):
        optimizer.zero_grad()
        
        # Compute dual objective
        dual_obj = epsilon * lambda_dual + (1.0 / N) * beta_vars.sum(dim=2)
        
        # Compute constraints
        constraint_violations = 0.0
        
        for s in range(N):
            sample_demands = historical_samples[:, s, :].unsqueeze(1).expand(-1, multi_width, -1)
            unvisited_mask = 1.0 - visited_nodes
            sample_penalty = (sample_demands * unvisited_mask * M).sum(dim=2)
            
            # Constraint: β_s >= sample_penalty - λ * ||ξ^s - ξ||_2
            # For simplicity, assume ξ = expected demand and ||ξ^s - ξ||_2 = 0
            constraint = beta_vars[:, :, s] - sample_penalty
            constraint_violations += torch.clamp(-constraint, min=0.0).sum()
        
        # Total loss (maximize dual objective while satisfying constraints)
        loss = -dual_obj.sum() + 1000.0 * constraint_violations  # Large penalty for constraint violations
        
        loss.backward()
        optimizer.step()
        
        # Project lambda to non-negative
        with torch.no_grad():
            lambda_dual.clamp_(min=0.0)
            beta_vars.clamp_(min=0.0)
    
    return lambda_dual.detach(), beta_vars.detach()


class DROObjective:
    """
    Class to handle DRO objective computation and optimization
    """
    
    def __init__(self, dro_params):
        self.dro_params = dro_params
        self.epsilon = dro_params['epsilon']
        self.M = dro_params['M']
        self.N = dro_params['N']
    
    def compute_routing_cost(self, solutions, coordinates):
        """
        Compute the routing cost component
        """
        batch_size, multi_width, tour_length = solutions.shape
        
        # Gather coordinates of visited nodes
        expanded_coords = coordinates.unsqueeze(1).expand(-1, multi_width, -1, -1)
        gathering_index = solutions.unsqueeze(-1).expand(-1, -1, -1, 2)
        ordered_coords = expanded_coords.gather(dim=2, index=gathering_index)
        
        # Compute distances between consecutive nodes
        rolled_coords = ordered_coords.roll(dims=2, shifts=-1)
        segment_lengths = ((ordered_coords - rolled_coords) ** 2).sum(3).sqrt()
        
        total_distances = segment_lengths.sum(2)
        return total_distances
    
    def compute_unvisited_penalty(self, visited_nodes, historical_samples):
        """
        Compute the penalty for unvisited nodes under uncertainty
        """
        batch_size, multi_width, problem_size = visited_nodes.shape
        N = historical_samples.shape[1]
        
        penalties = []
        
        for s in range(N):
            sample_demands = historical_samples[:, s, :].unsqueeze(1).expand(-1, multi_width, -1)
            unvisited_mask = 1.0 - visited_nodes
            penalty = (sample_demands * unvisited_mask * self.M).sum(dim=2)
            penalties.append(penalty)
        
        return torch.stack(penalties, dim=2)  # shape: (batch, multi, N)
    
    def compute_dro_objective(self, solutions, coordinates, visited_nodes, historical_samples):
        """
        Compute the full DRO objective
        """
        # Routing cost
        routing_cost = self.compute_routing_cost(solutions, coordinates)
        
        # Unvisited penalties
        unvisited_penalties = self.compute_unvisited_penalty(visited_nodes, historical_samples)
        
        # Solve for optimal dual variables (simplified)
        lambda_dual, beta_vars = solve_dual_problem(visited_nodes, historical_samples, self.dro_params)
        
        # DRO objective: routing_cost + ε*λ + (1/N)*Σβ_s
        dro_penalty = self.epsilon * lambda_dual + (1.0 / self.N) * beta_vars.sum(dim=2)
        
        total_objective = routing_cost + dro_penalty
        
        return total_objective, routing_cost, dro_penalty, lambda_dual, beta_vars


def evaluate_robustness(solutions, coordinates, base_demands, uncertainty_scenarios, capacity=1.0):
    """
    Evaluate the robustness of solutions under different uncertainty scenarios
    
    Args:
        solutions: (batch, multi, tour_length) - solution tours
        coordinates: (batch, nodes, 2) - node coordinates
        base_demands: (batch, customers) - base customer demands  
        uncertainty_scenarios: (batch, scenarios, customers) - different demand scenarios
        capacity: vehicle capacity
    
    Returns:
        robustness_metrics: dictionary with robustness statistics
    """
    batch_size, multi_width, tour_length = solutions.shape
    num_scenarios = uncertainty_scenarios.shape[1]
    
    scenario_costs = []
    scenario_feasibility = []
    
    for scenario in range(num_scenarios):
        scenario_demands = uncertainty_scenarios[:, scenario, :]
        
        # Check feasibility under this scenario
        feasible = check_feasible(solutions, scenario_demands, capacity)
        scenario_feasibility.append(feasible)
        
        # Compute cost (add large penalty for infeasible solutions)
        routing_costs = []
        for b in range(batch_size):
            for m in range(multi_width):
                tour = solutions[b, m, :]
                cost = 0.0
                
                for i in range(tour_length - 1):
                    current_node = tour[i].item()
                    next_node = tour[i + 1].item()
                    
                    # Distance between nodes
                    if current_node < coordinates.shape[1] and next_node < coordinates.shape[1]:
                        dist = torch.norm(coordinates[b, current_node] - coordinates[b, next_node]).item()
                        cost += dist
                
                # Add penalty for infeasibility
                if not feasible[b, m]:
                    cost += 1000.0  # Large penalty
                
                routing_costs.append(cost)
        
        scenario_costs.append(routing_costs)
    
    scenario_costs = torch.tensor(scenario_costs)  # shape: (scenarios, batch*multi)
    scenario_feasibility = torch.stack(scenario_feasibility, dim=2)  # shape: (batch, multi, scenarios)
    
    # Compute robustness metrics
    mean_cost = scenario_costs.mean(dim=0)
    max_cost = scenario_costs.max(dim=0)[0]
    cost_std = scenario_costs.std(dim=0)
    feasibility_rate = scenario_feasibility.float().mean(dim=2)
    
    robustness_metrics = {
        'mean_cost': mean_cost.reshape(batch_size, multi_width),
        'max_cost': max_cost.reshape(batch_size, multi_width),
        'cost_std': cost_std.reshape(batch_size, multi_width),
        'feasibility_rate': feasibility_rate,
        'scenario_costs': scenario_costs.reshape(num_scenarios, batch_size, multi_width),
        'scenario_feasibility': scenario_feasibility
    }
    
    return robustness_metrics