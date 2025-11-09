from dataclasses import dataclass
import torch
import numpy as np
import torch.nn.functional as F

from utils_dro import augment_xy_data_by_8_fold


@dataclass
class Reset_State:
    depot_xy: torch.Tensor = None
    # shape: (batch, 1, 2)
    node_xy: torch.Tensor = None
    # shape: (batch, problem, 2)
    node_demand: torch.Tensor = None
    # shape: (batch, problem)
    dist: torch.Tensor = None
    # shape: (batch, problem+1, problem+1)
    historical_samples: torch.Tensor = None
    # shape: (batch, N, problem) - historical demand samples


@dataclass
class Step_State:
    selected_count: int = None
    load: torch.Tensor = None
    # shape: (batch, multi)
    current_node: torch.Tensor = None
    # shape: (batch, multi)
    ninf_mask: torch.Tensor = None
    # shape: (batch, multi, problem+1)
    finished: torch.Tensor = None
    # shape: (batch, multi)
    visited_nodes: torch.Tensor = None
    # shape: (batch, multi, problem) - binary indicator for visited customer nodes


class CVRPEnv_DRO:
    def __init__(self, multi_width, device, dro_params=None):

        # Const @INIT
        ####################################
        self.device = device
        self.vrplib = False
        self.problem_size = None
        self.multi_width = multi_width

        self.depot_xy = None
        self.unscaled_depot_xy = None
        self.node_xy = None
        self.node_demand = None
        self.input_mask = None

        # DRO parameters
        ####################################
        if dro_params is None:
            dro_params = {
                "epsilon": 0.01,              # 从 0.1 改为 0.01
                "M": 10.0,                   # 从 1000.0 改为 10.0  
                "feasibility_penalty": 10.0,  # 从 1000.0 改为 10.0
                "robustness_weight": 0.01,    # 从 0.1 改为 0.01
                "uncertainty_std": 0.05,      # 从 0.15 改为 0.05
                "N": 10,                     # Number of samples
            }
        self.dro_params = dro_params
        
        self.historical_samples = None  # shape: (batch, N, problem)
        self.lambda_dual = None         # Dual variable λ
        self.beta_vars = None          # Variables β_s for each sample

        # Const @Load_Problem
        ####################################
        self.batch_size = None
        self.depot_node_xy = None
        # shape: (batch, problem+1, 2)
        self.depot_node_demand = None
        # shape: (batch, problem+1)

        # Dynamic-1
        ####################################
        self.selected_count = None
        self.current_node = None
        # shape: (batch, multi)
        self.selected_node_list = None
        # shape: (batch, multi, 0~)

        # Dynamic-2
        ####################################
        self.at_the_depot = None
        # shape: (batch, multi)
        self.load = None
        # shape: (batch, multi)
        self.visited_ninf_flag = None
        # shape: (batch, multi, problem+1)
        self.ninf_mask = None
        # shape: (batch, multi, problem+1)
        self.finished = None
        # shape: (batch, multi)
        self.visited_customers = None
        # shape: (batch, multi, problem) - binary indicator for visited customers

        # states to return
        ####################################
        self.reset_state = Reset_State()
        self.step_state = Step_State()
        self.empirical_distributions = []  # List to store empirical distributions
        self.use_empirical = False  # Flag to use empirical distribution

    def generate_historical_samples(self, base_demand, N=None):
        """Generate historical demand samples around base demand"""
        if N is None:
            N = self.dro_params['N']
        
        batch_size, problem_size = base_demand.shape
        
        # Generate samples with some noise around base demand
        # You can customize this based on your specific uncertainty model
        noise_std = 0.1  # 10% standard deviation
        samples = []
        
        for _ in range(N):
            # Add Gaussian noise
            noise = torch.randn_like(base_demand) * noise_std
            sample = torch.clamp(base_demand + noise, min=0.01, max=1.0)  # Keep demand positive and <= 1
            samples.append(sample)
        
        historical_samples = torch.stack(samples, dim=1)  # shape: (batch, N, problem)
        return historical_samples

    def load_vrplib_problem(self, instance, aug_factor=1, historical_samples=None):
        self.vrplib = True
        self.batch_size = 1
        node_coord = torch.FloatTensor(instance['node_coord']).unsqueeze(0).to(self.device)
        demand = torch.FloatTensor(instance['demand']).unsqueeze(0).to(self.device)
        demand = demand / instance['capacity']
        self.unscaled_depot_node_xy = node_coord
        # shape: (batch, problem+1, 2)
        
        min_x = torch.min(node_coord[:, :, 0], 1)[0]
        min_y = torch.min(node_coord[:, :, 1], 1)[0]
        max_x = torch.max(node_coord[:, :, 0], 1)[0]
        max_y = torch.max(node_coord[:, :, 1], 1)[0]
        scaled_depot_node_x = (node_coord[:, :, 0] - min_x) / (max_x - min_x)
        scaled_depot_node_y = (node_coord[:, :, 1] - min_y) / (max_y - min_y)
        
        self.depot_node_xy = torch.cat((scaled_depot_node_x[:, :, None]
                                        , scaled_depot_node_y[:, :, None]), dim=2)
        depot = self.depot_node_xy[:, instance['depot'], :]
        
        # Generate or use provided historical samples
        if historical_samples is None:
            self.historical_samples = self.generate_historical_samples(demand[:, 1:])
        else:
            self.historical_samples = historical_samples.to(self.device)
        
        if aug_factor > 1:
            if aug_factor == 8:
                self.batch_size = self.batch_size * 8
                depot = augment_xy_data_by_8_fold(depot)
                self.depot_node_xy = augment_xy_data_by_8_fold(self.depot_node_xy)
                self.unscaled_depot_node_xy = augment_xy_data_by_8_fold(self.unscaled_depot_node_xy)
                demand = demand.repeat(8, 1)
                self.historical_samples = self.historical_samples.repeat(8, 1, 1)
            else:
                raise NotImplementedError
        
        self.depot_node_demand = demand
        self.reset_state.depot_xy = depot
        self.reset_state.node_xy = self.depot_node_xy[:, 1:, :]
        self.reset_state.node_demand = demand[:, 1:]
        self.reset_state.historical_samples = self.historical_samples
        self.problem_size = self.reset_state.node_xy.shape[1]

        self.dist = (self.depot_node_xy[:, :, None, :] - self.depot_node_xy[:, None, :, :]).norm(p=2, dim=-1)
        # shape: (batch, problem+1, problem+1)
        self.reset_state.dist = self.dist

    def load_random_problems(self, batch, aug_factor=1, historical_samples=None):
        self.batch_size = batch['loc'].shape[0]
        node_coord = batch['loc'].to(self.device)
        demand = batch['demand'].to(self.device)
        depot = batch['depot'].to(self.device)
        if len(depot.shape) == 2:
            depot = depot[:, None, :]
            
        # Generate or use provided historical samples
        if historical_samples is None:
            self.historical_samples = self.generate_historical_samples(demand)
        else:
            self.historical_samples = historical_samples.to(self.device)
            
        if aug_factor > 1:
            if aug_factor == 8:
                self.batch_size = self.batch_size * 8
                depot = augment_xy_data_by_8_fold(depot)
                node_coord = augment_xy_data_by_8_fold(node_coord)
                demand = demand.repeat(8, 1)
                self.historical_samples = self.historical_samples.repeat(8, 1, 1)
            else:
                raise NotImplementedError
            
        self.depot_node_xy = torch.cat((depot, node_coord), dim=1)
        self.depot_node_demand = torch.cat((torch.zeros(self.batch_size, 1).to(self.device), demand), dim=1)    
            
        self.reset_state.depot_xy = depot
        self.reset_state.node_xy = self.depot_node_xy[:, 1:, :]
        self.reset_state.node_demand = demand
        self.reset_state.historical_samples = self.historical_samples
        self.problem_size = self.reset_state.node_xy.shape[1]
        self.dist = (self.depot_node_xy[:, :, None, :] - self.depot_node_xy[:, None, :, :]).norm(p=2, dim=-1)
        # shape: (batch, problem+1, problem+1)
        self.reset_state.dist = self.dist

    def reset(self):
        self.selected_count = 0
        self.current_node = None
        # shape: (batch, multi)
        self.selected_node_list = torch.zeros(size=(self.batch_size, self.multi_width, 0), dtype=torch.long, device=self.device)
        # shape: (batch, multi, 0~)

        self.at_the_depot = torch.ones(size=(self.batch_size, self.multi_width), dtype=torch.bool, device=self.device)
        # shape: (batch, multi)
        self.load = torch.ones(size=(self.batch_size, self.multi_width), device=self.device)
        # shape: (batch, multi)
        self.visited_ninf_flag = torch.zeros(size=(self.batch_size, self.multi_width, self.problem_size+1), device=self.device)
        # shape: (batch, multi, problem+1)
        if self.input_mask is not None:
            self.visited_ninf_flag = self.input_mask[:, None, :].expand(self.batch_size, self.multi_width, self.problem_size+1).clone()
        self.ninf_mask = torch.zeros(size=(self.batch_size, self.multi_width, self.problem_size+1), device=self.device)
        # shape: (batch, multi, problem+1)
        self.finished = torch.zeros(size=(self.batch_size, self.multi_width), dtype=torch.bool, device=self.device)
        # shape: (batch, multi)
        
        # Initialize visited customers tracking (excluding depot)
        self.visited_customers = torch.zeros(size=(self.batch_size, self.multi_width, self.problem_size), device=self.device)
        # shape: (batch, multi, problem)

        # Initialize dual variables
        self.lambda_dual = torch.zeros(size=(self.batch_size, self.multi_width), device=self.device)
        self.beta_vars = torch.zeros(size=(self.batch_size, self.multi_width, self.dro_params['N']), device=self.device)

        reward = None
        done = False
        return self.reset_state, reward, done

    def reset_width(self, new_width):
        self.multi_width = new_width

    def pre_step(self):
        self.step_state.selected_count = self.selected_count
        self.step_state.load = self.load
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished
        self.step_state.visited_nodes = self.visited_customers

        reward = None
        done = False
        return self.step_state, reward, done

    def step(self, selected):
        # selected.shape: (batch, multi)
        # Dynamic-1
        ####################################
        
        self.selected_count += 1
        self.current_node = selected
        # shape: (batch, multi)
        self.selected_node_list = torch.cat((self.selected_node_list, self.current_node[:, :, None]), dim=2)
        # shape: (batch, multi, 0~)

        # Update visited customers (excluding depot node 0)
        customer_mask = selected > 0  # True for customer nodes (not depot)
        customer_indices = selected - 1  # Convert to 0-indexed for customers
        customer_indices = torch.clamp(customer_indices, min=0)  # Ensure non-negative
        
        # Update visited customers
        for b in range(self.batch_size):
            for m in range(self.multi_width):
                if customer_mask[b, m]:
                    self.visited_customers[b, m, customer_indices[b, m]] = 1.0

        # Dynamic-2
        ####################################
        self.at_the_depot = (selected == 0)

        demand_list = self.depot_node_demand[:, None, :].expand(self.batch_size, self.multi_width, -1)
        # shape: (batch, multi, problem+1)
        gathering_index = selected[:, :, None]
        # shape: (batch, multi, 1)
        selected_demand = demand_list.gather(dim=2, index=gathering_index).squeeze(dim=2)
        # shape: (batch, multi)
        self.load -= selected_demand
        self.load[self.at_the_depot] = 1 # refill loaded at the depot

        self.visited_ninf_flag.scatter_(2, self.selected_node_list, float('-inf'))
        # shape: (batch, multi, problem+1)
        self.visited_ninf_flag[:, :, 0][~self.at_the_depot] = 0  # depot is considered unvisited, unless you are AT the depot

        self.ninf_mask = self.visited_ninf_flag.clone()
        round_error_epsilon = 1e-6
        demand_too_large = self.load[:, :, None] + round_error_epsilon < demand_list
        # shape: (batch, multi, problem+1)
        self.ninf_mask[demand_too_large] = float('-inf')
        # shape: (batch, multi, problem+1)

        newly_finished = (self.visited_ninf_flag == float('-inf')).all(dim=2)
        # shape: (batch, multi)
        self.finished = self.finished + newly_finished
        # shape: (batch, multi)

        # do not mask depot for finished episode.
        self.ninf_mask[:, :, 0][self.finished] = 0

        self.step_state.selected_count = self.selected_count
        self.step_state.load = self.load
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished
        self.step_state.visited_nodes = self.visited_customers
        
        # returning values
        done = self.finished.all()
        if done:
            if self.vrplib == True:
                reward = self.compute_unscaled_dro_reward()
            else:
                reward = self._get_dro_reward()
        else:
            reward = None

        return self.step_state, reward, done

    def _get_dro_reward(self):
        """Compute DRO reward including routing cost and robust penalty"""
        # Original routing cost
        gathering_index = self.selected_node_list[:, :, :, None].expand(-1, -1, -1, 2)
        # shape: (batch, multi, selected_list_length, 2)
        all_xy = self.depot_node_xy[:, None, :, :].expand(-1, self.multi_width, -1, -1)
        # shape: (batch, multi, problem+1, 2)

        ordered_seq = all_xy.gather(dim=2, index=gathering_index)
        # shape: (batch, multi, selected_list_length, 2)

        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        segment_lengths = ((ordered_seq-rolled_seq)**2).sum(3).sqrt()
        # shape: (batch, multi, selected_list_length)

        travel_distances = segment_lengths.sum(2)
        # shape: (batch, multi)

        # DRO penalty term
        dro_penalty = self.compute_dro_penalty()
        
        # Total objective: minimize travel distance + DRO penalty
        total_cost = travel_distances + dro_penalty
        
        return -total_cost  # Negative because we want to minimize

    def add_empirical_distribution(self, empirical_dist):
        """
        Add empirical distribution to the stored distributions
        
        Args:
            empirical_dist: torch.Tensor of shape (batch, multi, problem) 
                           Binary indicator where 1 means node is visited, 0 means unvisited
                           Length equals current node sequence count
        """
        if not isinstance(empirical_dist, torch.Tensor):
            empirical_dist = torch.tensor(empirical_dist, device=self.device)
        
        # Ensure the empirical distribution has the correct shape
        if len(empirical_dist.shape) == 1:
            # If 1D, expand to (batch, multi, problem)
            empirical_dist = empirical_dist.unsqueeze(0).unsqueeze(0).expand(
                self.batch_size, self.multi_width, -1)
        elif len(empirical_dist.shape) == 2:
            # If 2D, expand to (batch, multi, problem)
            empirical_dist = empirical_dist.unsqueeze(1).expand(-1, self.multi_width, -1)
        
        # Store the empirical distribution
        self.empirical_distributions.append(empirical_dist.clone())
        self.use_empirical = True
        
        print(f"Added empirical distribution with shape: {empirical_dist.shape}")
        print(f"Total empirical distributions stored: {len(self.empirical_distributions)}")

    def get_empirical_distributions(self):
        """
        Get all stored empirical distributions
        
        Returns:
            List of empirical distributions or None if no distributions stored
        """
        if not self.empirical_distributions:
            return None
        return self.empirical_distributions

    def clear_empirical_distributions(self):
        """Clear all stored empirical distributions"""
        self.empirical_distributions = []
        self.use_empirical = False
        print("Cleared all empirical distributions")

    def compute_dro_penalty(self):
        """
        Compute the DRO penalty term: ε*λ + (1/N)*Σβ_s
        Now supports both historical samples and empirical distributions
        """
        epsilon = self.dro_params['epsilon']
        M = self.dro_params['M']
        N = self.dro_params['N']
        
        # Determine which distributions to use
        if self.use_empirical and self.empirical_distributions:
            # Use empirical distributions
            distributions = self.empirical_distributions
            N_effective = len(distributions)
            print(f"Using {N_effective} empirical distributions for DRO penalty computation")
        else:
            # Use historical samples (original behavior)
            distributions = None
            N_effective = N
        
        # Update dual variables (simplified dual update)
        unvisited_penalties = []
        
        if distributions is not None:
            # Use empirical distributions
            for i, emp_dist in enumerate(distributions):
                # emp_dist is already a binary indicator (1 for visited, 0 for unvisited)
                # We need unvisited mask, so take complement
                unvisited_mask = 1.0 - emp_dist  # shape: (batch, multi, problem)
                
                # For empirical distributions, we can use current demands or average demands
                if hasattr(self, 'current_sample'):
                    current_demands = self.current_sample.unsqueeze(1).expand(-1, self.multi_width, -1)
                else:
                    # Use average demand from historical samples
                    avg_demands = self.historical_samples.mean(dim=1).unsqueeze(1).expand(-1, self.multi_width, -1)
                    current_demands = avg_demands
                
                # Penalty for unvisited nodes
                sample_penalty = (current_demands * unvisited_mask * M).sum(dim=2)  # shape: (batch, multi)
                
                # Compute distance between current empirical distribution and this stored distribution
                if i < len(self.empirical_distributions) - 1:
                    # Distance to other empirical distributions
                    dist_distance = torch.norm(emp_dist - self.visited_customers, p=2, dim=2)
                else:
                    # Distance for current distribution
                    dist_distance = torch.zeros_like(sample_penalty)
                
                # Update beta_s (dual variable for empirical distribution i)
                if i < self.beta_vars.shape[2]:
                    constraint_value = sample_penalty - self.lambda_dual * dist_distance
                    self.beta_vars[:, :, i] = torch.clamp(constraint_value, min=0.0)
                
                unvisited_penalties.append(sample_penalty)
        
        else:
            # Original historical samples approach
            for s in range(N_effective):
                # For sample s, compute penalty for unvisited nodes
                sample_demands = self.historical_samples[:, s, :].unsqueeze(1)  # shape: (batch, 1, problem)
                sample_demands = sample_demands.expand(-1, self.multi_width, -1)  # shape: (batch, multi, problem)
                
                # Penalty for unvisited nodes
                unvisited_mask = 1.0 - self.visited_customers  # shape: (batch, multi, problem)
                sample_penalty = (sample_demands * unvisited_mask * M).sum(dim=2)  # shape: (batch, multi)
                
                # Compute distance between current sample and historical sample s
                if hasattr(self, 'current_sample'):
                    sample_distance = torch.norm(self.current_sample.unsqueeze(1).expand(-1, self.multi_width, -1) - 
                                               sample_demands, p=2, dim=2)
                else:
                    sample_distance = torch.zeros_like(sample_penalty)
                
                # Update beta_s (dual variable for sample s)
                constraint_value = sample_penalty - self.lambda_dual * sample_distance
                self.beta_vars[:, :, s] = torch.clamp(constraint_value, min=0.0)
                
                unvisited_penalties.append(sample_penalty)
        
        # Compute DRO objective components
        lambda_term = epsilon * self.lambda_dual
        
        # Use effective number of distributions
        if distributions is not None:
            beta_term = self.beta_vars[:, :, :len(distributions)].mean(dim=2)  # Average over empirical distributions
        else:
            beta_term = self.beta_vars.mean(dim=2)  # Average over samples
        
        dro_penalty = lambda_term + beta_term
        return dro_penalty

    def update_empirical_distribution_from_current_state(self):
        """
        Update empirical distribution based on current visited customers state
        This can be called during the solving process to capture current visiting pattern
        """
        current_empirical = self.visited_customers.clone()  # shape: (batch, multi, problem)
        self.add_empirical_distribution(current_empirical)
        return current_empirical

    def get_empirical_statistics(self):
        """
        Get statistics about stored empirical distributions
        
        Returns:
            Dictionary with statistics
        """
        if not self.empirical_distributions:
            return {"count": 0, "avg_visited_rate": 0.0}
        
        # Stack all distributions
        all_dists = torch.stack(self.empirical_distributions, dim=0)  # shape: (num_dists, batch, multi, problem)
        
        # Compute statistics
        avg_visited_rate = all_dists.mean().item()
        visit_variance = all_dists.var().item()
        
        return {
            "count": len(self.empirical_distributions),
            "avg_visited_rate": avg_visited_rate,
            "visit_variance": visit_variance,
            "shape": all_dists.shape
        }

    def compute_unscaled_dro_reward(self, solutions=None, rounding=True):
        """Compute unscaled DRO reward for VRPLib instances"""
        if solutions is None:
            solutions = self.selected_node_list
            
        # Original routing cost (unscaled)
        gathering_index = solutions[:, :, :, None].expand(-1, -1, -1, 2)
        # shape: (batch, multi, selected_list_length, 2)
        all_xy = self.unscaled_depot_node_xy[:, None, :, :].expand(-1, self.multi_width, -1, -1)
        # shape: (batch, multi, problem+1, 2)

        ordered_seq = all_xy.gather(dim=2, index=gathering_index)
        # shape: (batch, multi, selected_list_length, 2)

        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)

        segment_lengths = ((ordered_seq-rolled_seq)**2).sum(3).sqrt()
        if rounding == True:
            segment_lengths = torch.round(segment_lengths)
        # shape: (batch, multi, selected_list_length)

        travel_distances = segment_lengths.sum(2)
        # shape: (batch, multi)
        
        # DRO penalty term (scaled appropriately)
        dro_penalty = self.compute_dro_penalty()   # Scale factor for unscaled problems
        
        total_cost = travel_distances + dro_penalty
        return -total_cost

    def get_cur_feature(self):
        if self.current_node is None:
            return None, None, None, None, None
        
        current_node = self.current_node[:, :, None, None].expand(self.batch_size, self.multi_width, 1, self.problem_size + 1)

        # Compute the relative distance
        cur_dist = torch.take_along_dim(self.dist[:, None, :, :].expand(self.batch_size, self.multi_width, self.problem_size + 1, self.problem_size + 1), 
                                        current_node, dim=2).squeeze(2)
        # shape: (batch, multi, problem)
        
        expanded_xy = self.depot_node_xy[:, None, :, :].expand(self.batch_size, self.multi_width, self.problem_size + 1, 2)
        relative_xy = expanded_xy - torch.take_along_dim(expanded_xy, self.current_node[:, :, None, None].expand(
            self.batch_size, self.multi_width, 1, 2), dim=2)
        # shape: (batch, problem, 2)

        relative_x = relative_xy[:, :, :, 0]
        relative_y = relative_xy[:, :, :, 1]

        # Compute the relative coordinates
        cur_theta = torch.atan2(relative_y, relative_x)
        # shape: (batch, multi, problem)

        # Compute the normalized demand. inf generated by division will be masked. 
        demand_list = self.depot_node_demand[:, None, :].expand(self.batch_size, self.multi_width, -1)
        norm_demand = demand_list / self.load[:, :, None]

        # Return visited nodes information as well
        return cur_dist, cur_theta, relative_xy, norm_demand, self.visited_customers
    
    def _get_routing_cost_only(self):
        """Compute only the routing cost without DRO penalty"""
        gathering_index = self.selected_node_list[:, :, :, None].expand(-1, -1, -1, 2)
        #print(gathering_index)
        # shape: (batch, multi, selected_list_length, 2)
        all_xy = self.depot_node_xy[:, None, :, :].expand(-1, self.multi_width, -1, -1)
        # shape: (batch, multi, problem+1, 2)

        ordered_seq = all_xy.gather(dim=2, index=gathering_index)
        
        # shape: (batch, multi, selected_list_length, 2)

        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        print(rolled_seq)
        segment_lengths = ((ordered_seq-rolled_seq)**2).sum(3).sqrt()
        # shape: (batch, multi, selected_list_length)

        travel_distances = segment_lengths.sum(2)
        # shape: (batch, multi)
        return -travel_distances

    def compute_unscaled_routing_cost_only(self, solutions=None, rounding=True):
        """Compute unscaled routing cost without DRO penalty for VRPLib instances"""
        if solutions is None:
            solutions = self.selected_node_list
            
        gathering_index = solutions[:, :, :, None].expand(-1, -1, -1, 2)
        # shape: (batch, multi, selected_list_length, 2)
        all_xy = self.unscaled_depot_node_xy[:, None, :, :].expand(-1, self.multi_width, -1, -1)
        # shape: (batch, multi, problem+1, 2)

        ordered_seq = all_xy.gather(dim=2, index=gathering_index)
        # shape: (batch, multi, selected_list_length, 2)

        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)

        segment_lengths = ((ordered_seq-rolled_seq)**2).sum(3).sqrt()
        if rounding == True:
            segment_lengths = torch.round(segment_lengths)
        # shape: (batch, multi, selected_list_length)

        travel_distances = segment_lengths.sum(2)
        # shape: (batch, multi)
        return -travel_distances