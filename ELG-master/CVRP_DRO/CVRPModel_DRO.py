import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math

from models import *
from models import _get_encoding


class CVRPModel_DRO(nn.Module):

    def __init__(self, **model_params):
        nn.Module.__init__(self)
        self.model_params = model_params

        self.encoder = CVRP_Encoder_DRO(**model_params)
        self.decoder = CVRP_Decoder_DRO(**model_params)
        self.encoded_nodes = None
        # shape: (batch, problem, embedding)

    def pre_forward(self, reset_state):
        depot_xy = reset_state.depot_xy
        # shape: (batch, 1, 2)
        node_xy = reset_state.node_xy
        # shape: (batch, problem, 2)
        node_demand = reset_state.node_demand
        # shape: (batch, problem)
        dist = reset_state.dist
        # shape: (batch, problem+1, problem+1)
        historical_samples = reset_state.historical_samples
        # shape: (batch, N, problem)
        
        node_xy_demand = torch.cat((node_xy, node_demand[:, :, None]), dim=2)
        # shape: (batch, problem, 3)
        
        self.encoded_nodes = self.encoder(depot_xy, node_xy_demand, dist, historical_samples)
        # shape: (batch, problem+1, embedding)
        self.decoder.set_kv(self.encoded_nodes)

    def one_step_rollout(self, state, cur_dist, cur_theta, xy, norm_demand, visited_nodes, eval_type):
        device = state.ninf_mask.device
        batch_size = state.ninf_mask.shape[0]
        multi_width = state.ninf_mask.shape[1]
        problem_size = state.ninf_mask.shape[2] - 1
        
        if state.selected_count == 0:  # First Move, depot
            selected = torch.zeros(size=(batch_size, multi_width), dtype=torch.long, device=device)
            prob = torch.ones(size=(batch_size, multi_width), device=device)

        elif state.selected_count == 1:  # Second Move, POMO
            selected = torch.tensor(random.sample(range(0, problem_size), multi_width), device=device)[
                           None, :] \
                    .expand(batch_size, multi_width)
            # shape: (batch, pomo+1)
            prob = torch.ones(size=(batch_size, multi_width), device=device)

        else:
            encoded_last_node = _get_encoding(self.encoded_nodes, state.current_node)
            # shape: (batch, pomo+1, embedding)
            probs = self.decoder(encoded_last_node, state.load, cur_dist, cur_theta, xy, 
                               norm_demand=norm_demand, ninf_mask=state.ninf_mask)
            # shape: (batch, pomo+1, problem+1)

            if eval_type == 'sample':
                with torch.no_grad():
                    selected = probs.reshape(batch_size * multi_width, -1).multinomial(1) \
                        .squeeze(dim=1).reshape(batch_size, multi_width)
                # shape: (batch, pomo+1)
                prob = torch.take_along_dim(probs, selected[:, :, None], dim=2).reshape(batch_size, multi_width)
                # shape: (batch, pomo+1)
                if not (prob != 0).all():   # avoid sampling prob 0
                    prob += 1e-6

            else:
                selected = probs.argmax(dim=2)
                # shape: (batch, pomo+1)
                prob = None  # value not needed. Can be anything.

        return selected, prob


class CVRP_Encoder_DRO(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        encoder_layer_num = self.model_params['encoder_layer_num']

        self.embedding_depot = nn.Linear(2, embedding_dim)
        self.embedding_node = nn.Linear(3, embedding_dim)
        
        # Additional embedding for uncertainty information
        self.uncertainty_embedding = nn.Linear(model_params.get('N', 50), embedding_dim // 4)  # Embed historical samples
        self.feature_combine = nn.Linear(embedding_dim + embedding_dim // 4, embedding_dim)
        
        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(encoder_layer_num)])

    def forward(self, depot_xy, node_xy_demand, dist, historical_samples):
        # depot_xy.shape: (batch, 1, 2)
        # node_xy_demand.shape: (batch, problem, 3)
        # dist.shape: (batch, problem+1, problem+1)
        # historical_samples.shape: (batch, N, problem)

        embedded_depot = self.embedding_depot(depot_xy)
        # shape: (batch, 1, embedding)
        
        embedded_node = self.embedding_node(node_xy_demand)
        # shape: (batch, problem, embedding)

        # Embed uncertainty information
        # Compute statistics over historical samples for each node
        uncertainty_features = historical_samples.transpose(1, 2)  # shape: (batch, problem, N)
        embedded_uncertainty = self.uncertainty_embedding(uncertainty_features)
        # shape: (batch, problem, embedding//4)
        
        # Combine node features with uncertainty features
        combined_node_features = torch.cat([embedded_node, embedded_uncertainty], dim=-1)
        combined_node_features = self.feature_combine(combined_node_features)
        # shape: (batch, problem, embedding)
        
        # For depot, just use zero uncertainty embedding
        depot_uncertainty = torch.zeros(embedded_depot.shape[0], 1, embedded_uncertainty.shape[-1], 
                                      device=embedded_depot.device)
        combined_depot_features = torch.cat([embedded_depot, depot_uncertainty], dim=-1)
        combined_depot_features = self.feature_combine(combined_depot_features)
        # shape: (batch, 1, embedding)

        out = torch.cat((combined_depot_features, combined_node_features), dim=1)
        # shape: (batch, problem+1, embedding)

        for layer in self.layers:
            out = layer(out)

        return out
        # shape: (batch, problem+1, embedding)


class CVRP_Decoder_DRO(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        # 恢复到原始维度 (embedding_dim + 1，只包含load信息)
        self.Wq_last = nn.Linear(embedding_dim+1, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)

        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.k = None  # saved key, for multi-head attention
        self.v = None  # saved value, for multi-head_attention
        self.single_head_key = None  # saved, for single-head attention

        self.local = False
    
    def add_local_policy(self, device):
        self.local_policies = nn.ModuleList([local_policy_att_dro(self.model_params, idx=i).to(device) for i in range(self.model_params['ensemble_size'])])
        self.local = True

    def set_kv(self, encoded_nodes):
        # encoded_nodes.shape: (batch, problem+1, embedding)
        head_num = self.model_params['head_num']

        self.k = reshape_by_heads(self.Wk(encoded_nodes), head_num=head_num)
        self.v = reshape_by_heads(self.Wv(encoded_nodes), head_num=head_num)
        # shape: (batch, head_num, problem+1, qkv_dim)
        self.single_head_key = encoded_nodes.transpose(1, 2)
        # shape: (batch, embedding, problem+1)

    def forward(self, encoded_last_node, load, cur_dist, cur_theta, xy, norm_demand, ninf_mask):
        # encoded_last_node.shape: (batch, pomo, embedding)
        # load.shape: (batch, pomo)
        # ninf_mask.shape: (batch, pomo, problem)
        head_num = self.model_params['head_num']

        #  Multi-Head Attention
        #######################################################
        # 只使用encoded_last_node和load，移除visited_nodes相关代码
        input_cat = torch.cat((encoded_last_node, load[:, :, None]), dim=2)
        # shape = (batch, group, EMBEDDING_DIM+1)

        q_last = reshape_by_heads(self.Wq_last(input_cat), head_num=head_num)
        # shape: (batch, head_num, pomo, qkv_dim)

        q = q_last
        # shape: (batch, head_num, pomo, qkv_dim)
        out_concat = multi_head_attention(q, self.k, self.v, rank3_ninf_mask=ninf_mask)
        # shape: (batch, pomo, head_num*qkv_dim)

        mh_atten_out = self.multi_head_combine(out_concat)
        # shape: (batch, pomo, embedding)

        #  Single-Head Attention, for probability calculation
        #######################################################
        score = torch.matmul(mh_atten_out, self.single_head_key)
        # shape: (batch, pomo, problem)

        sqrt_embedding_dim = self.model_params['embedding_dim'] ** 0.5
        logit_clipping = self.model_params['logit_clipping']

        score_scaled = score / sqrt_embedding_dim

        # DRO-specific modifications (简化版本)
        if self.model_params.get('dro_penalty_in_decoder', False):
            # 简化的uncertainty penalty，不依赖visited_nodes
            uncertainty_penalty = self._compute_uncertainty_penalty(cur_dist)
            score_scaled += uncertainty_penalty

        if self.model_params['distance_penalty']:
            local_size = self.model_params['local_size'][0]
            valid_nodes = cur_dist.shape[2]
            dist = cur_dist.clone()
            # Unselected neighbor nodes except depot (depot will be added below)
            mask = ninf_mask.clone()
            # mask depot
            mask[:, :, 0] = float('-inf')
            dist -= mask
            valid_nodes -= mask.isinf().sum(-1).min()

            if local_size > valid_nodes:
                local_size = valid_nodes
            else:
                local_size = local_size
            
            # Zero idx for depot
            depot_idx = torch.zeros(dist.shape[0], dist.shape[1], 1, device=dist.device).long()

            if local_size > 0:
                # Topk except depot, but the idx should add 1 for alignment
                dist_, idx = dist[:, :, 1:].topk(local_size, dim=-1, largest=False)
                # norm factor is computed before depot is added
                if dist_.isinf().any():
                    dist_[dist_.isinf()] = 0.
                norm_idx = dist_.max(-1)[0] != 0
                norm_fac = dist_[norm_idx].max(-1)[0].unsqueeze(-1)

                idx += 1
                # shape: (batch, multi, local)
                # Add depot idx
                idx = torch.cat((depot_idx, idx), dim=-1)
            else:
                # No other nodes can be selected except depot
                idx = depot_idx
                norm_idx = None

            sorted_dist = torch.take_along_dim(dist, idx, dim=-1)
            # shape: (batch, multi, local)

            # Padding 0
            # Check if there are some dims that require padding
            if sorted_dist.isinf().any():
                sorted_dist[sorted_dist.isinf()] = 0.
            
            if norm_idx is None:
                norm_idx = sorted_dist.max(-1)[0] != 0
                sorted_dist[norm_idx] = sorted_dist[norm_idx] / sorted_dist[norm_idx].max(-1)[0].unsqueeze(-1)
            else:
                sorted_dist[norm_idx] = sorted_dist[norm_idx] / norm_fac

            dist_penalty = - sorted_dist
            out_mat = self.model_params['xi'] * torch.ones(cur_dist.shape, device=cur_dist.device)
            score_scaled += out_mat.scatter_(-1, idx, dist_penalty)

        if self.model_params['ensemble'] and self.local:
            score_local = 0.
            for i in range(self.model_params['ensemble_size']):
                score_local += self.local_policies[i](theta=cur_theta, dist=cur_dist, xy=xy, 
                                                    norm_demand=norm_demand, ninf_mask=ninf_mask)
            score_scaled += score_local / self.model_params['ensemble_size']
            # shape: (batch, pomo, problem)

        score_clipped = logit_clipping * torch.tanh(score_scaled)

        score_masked = score_clipped + ninf_mask

        probs = F.softmax(score_masked, dim=2)
        # shape: (batch, pomo, problem)

        return probs
    
    def _compute_uncertainty_penalty(self, cur_dist):
        """简化的uncertainty penalty，不依赖visited_nodes"""
        # 简单地对距离较远的节点施加小的惩罚
        penalty = cur_dist * 0.05  # 小的距离惩罚因子
        return penalty


class local_policy_att_dro(nn.Module):
    def __init__(self, model_params, idx=0):
        super().__init__()
        self.emb_dim = model_params['local_att_hidden_dim']
        self.head_num = model_params['local_att_head_num']
        self.qkv_dim = model_params['local_att_qkv_dim']
        self.model_params = model_params
        self.local_size = model_params['local_size'][idx]
        
        # 恢复到原始维度，不包含visited_nodes信息
        if model_params['demand']:
            self.init_emb = nn.Linear(3, self.emb_dim)  # 距离、角度、需求
        else:
            self.init_emb = nn.Linear(2, self.emb_dim)  # 距离、角度
            
        self.cur_token_emb = nn.Parameter(torch.Tensor(self.emb_dim))
        self.cur_token_emb.data.uniform_(-1, 1)
        self.Wq = nn.Linear(self.emb_dim, self.head_num * self.qkv_dim, bias=False)
        self.Wk = nn.Linear(self.emb_dim, self.head_num * self.qkv_dim, bias=False)
        self.Wv = nn.Linear(self.emb_dim, self.head_num * self.qkv_dim, bias=False)

        self.multi_head_combine = nn.Linear(self.head_num * self.qkv_dim, self.emb_dim)

        # For positional encoding
        num_timescales = self.emb_dim // 2
        max_timescale = 10000.0
        min_timescale = 1.0
        log_timescale_increment = (
            math.log(float(max_timescale) / float(min_timescale)) /
            max(num_timescales - 1, 1))
        self.inv_timescales = min_timescale * torch.exp(
            torch.arange(num_timescales, dtype=torch.float32) *
            -log_timescale_increment)

    def get_position_encoding(self, x):
        self.inv_timescales = self.inv_timescales.to(x.device)

        max_length = x.size()[2]
        position = torch.arange(max_length, dtype=torch.float32,
                                device=x.device)
        scaled_time = position.unsqueeze(1) * self.inv_timescales.unsqueeze(0)
        signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)],
                           dim=1)
        signal = F.pad(signal, (0, 0, 0, self.emb_dim % 2))
        signal = signal.view(1, max_length, self.emb_dim)
        return signal

    def forward(self, theta, dist, xy, norm_demand=None, ninf_mask=None):
        # 移除visited_nodes参数，其他逻辑保持不变
        valid_nodes = dist.shape[2]
        multi_width = dist.shape[1]
        # Unselected neighbor nodes except depot (depot will be added below)
        mask = ninf_mask.clone()
        # mask depot
        mask[:, :, 0] = float('-inf')
        dist -= mask
        valid_nodes -= mask.isinf().sum(-1).min()

        if self.local_size > valid_nodes:
            local_size = valid_nodes
        else:
            local_size = self.local_size
        
        # Zero idx for depot
        depot_idx = torch.zeros(dist.shape[0], dist.shape[1], 1, device=dist.device).long()

        if local_size > 0:
            # Topk except depot, but the idx should add 1 for alignment
            dist_, idx = dist[:, :, 1:].topk(local_size, dim=-1, largest=False)
            # norm factor is computed before depot is added
            if dist_.isinf().any():
                dist_[dist_.isinf()] = 0.
            norm_idx = dist_.max(-1)[0] != 0
            norm_fac = dist_[norm_idx].max(-1)[0].unsqueeze(-1) + 1e-6    # avoid division by zero

            idx += 1
            # shape: (batch, multi, local)
            # Add depot idx
            idx = torch.cat((depot_idx, idx), dim=-1)
        else:
            # No other nodes can be selected except depot
            idx = depot_idx
            norm_idx = None

        sorted_dist = torch.take_along_dim(dist, idx, dim=-1)
        sorted_theta = torch.take_along_dim(theta, idx, dim=-1)
        sorted_demand = torch.take_along_dim(norm_demand, idx, dim=-1)
        sorted_mask = torch.take_along_dim(ninf_mask, idx, dim=-1)
        
        # shape: (batch, multi, local)
        if self.model_params['euclidean'] == True:
            sorted_x = torch.take_along_dim(xy[:, :, :, 0], idx, dim=-1)
            sorted_y = torch.take_along_dim(xy[:, :, :, 1], idx, dim=-1)

        # Padding 0
        # Check if there are some dims that require padding
        if sorted_dist.isinf().any():
            sorted_theta[sorted_dist.isinf()] = 0.
            sorted_demand[sorted_dist.isinf()] = 0.
            if self.model_params['euclidean'] == True:
                sorted_x[sorted_dist.isinf()] = 0.
                sorted_y[sorted_dist.isinf()] = 0.
            sorted_dist[sorted_dist.isinf()] = 0.
        
        if norm_idx is None:
            norm_idx = sorted_dist.max(-1)[0] != 0
            norm_fac = sorted_dist[norm_idx].max(-1)[0].unsqueeze(-1) + 1e-6    # avoid division by zero
            sorted_dist[norm_idx] = sorted_dist[norm_idx] / norm_fac
            if self.model_params['euclidean'] == True:
                sorted_x[norm_idx] = sorted_x[norm_idx] / norm_fac
                sorted_y[norm_idx] = sorted_y[norm_idx] / norm_fac
        else:
            sorted_dist[norm_idx] = sorted_dist[norm_idx] / norm_fac
            if self.model_params['euclidean'] == True:
                sorted_x[norm_idx] = sorted_x[norm_idx] / norm_fac
                sorted_y[norm_idx] = sorted_y[norm_idx] / norm_fac

        if self.model_params['euclidean'] == True:
            sorted_dist_theta = torch.cat((sorted_x[:, :, :, None], sorted_y[:, :, :, None]), dim=-1)
        else:
            sorted_dist_theta = torch.cat((sorted_dist[:, :, :, None], sorted_theta[:, :, :, None]), dim=-1)
        # shape: (batch, multi, local, 2)
        
        # 不包含visited_nodes信息
        if self.model_params['demand']:
            sorted_input = torch.cat((sorted_dist_theta, sorted_demand[:, :, :, None]), dim=-1)
            # shape: (batch, multi, local, 3)
        else:
            sorted_input = sorted_dist_theta
        
        cur_token = self.cur_token_emb[None, None, :].expand(dist.shape[0], dist.shape[1], self.emb_dim)
        # shape: (batch, multi, emb)

        if self.model_params['positional']:
            # Positional embedding
            signal = self.get_position_encoding(sorted_input)[:, None, :, :].expand(-1, multi_width, -1, -1)
            init_k = self.init_emb(sorted_input) + signal
        else:
            init_k = self.init_emb(sorted_input)
            # shape: (batch, multi, local, emb) 

        q = reshape_by_heads(self.Wq(cur_token), head_num=self.head_num).unsqueeze(3)
        # shape: (batch, head_num, multi, 1, qkv_dim)
        k = reshape_by_heads(self.Wk(init_k), head_num=self.head_num)
        # shape: (batch, head_num, multi, local, qkv_dim)
        v = reshape_by_heads(self.Wv(init_k), head_num=self.head_num)
        # shape: (batch, head_num, multi, local, qkv_dim)

        out_concat = multi_head_attention(q, k, v, rank3_ninf_mask=sorted_mask)
        # shape: (batch, multi, head_num * qkv_dim)
        
        mh_atten_out = self.multi_head_combine(out_concat).unsqueeze(2)
        # shape: (batch, multi, 1, emb)

        score = torch.matmul(mh_atten_out, init_k.transpose(2, 3)).squeeze(2)
        # shape: (batch, multi, local)
        
        sqrt_emb_dim = self.emb_dim ** 0.5
        score_scaled = score / sqrt_emb_dim

        out = score_scaled
        out_mat = torch.zeros(dist.shape, device=dist.device)
        # shape: (batch, multi, problem+1)
        
        out = out_mat.scatter_(-1, idx, out)
        # shape: (batch, multi, problem+1)

        return out