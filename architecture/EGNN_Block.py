#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   EGNN Block
@Time    :   2022/09/05 10:57:54
@Author  :   Xujun Zhang
@Version :   1.0
@Contact :   
@License :   
@Desc    :   None
'''

# here put the import lib
import math
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GraphNorm
from torch_geometric.utils import softmax, to_dense_batch
from torch_scatter import scatter, scatter_mean

class EGNN(nn.Module):
    def __init__(self, dim_in, dim_tmp, edge_in, edge_out, num_head=8, drop_rate=0.15):
        super().__init__()
        assert dim_tmp % num_head == 0
        self.edge_dim = edge_in
        self.num_head = num_head # 4
        self.dh = dim_tmp // num_head # 32
        self.dim_tmp = dim_tmp # 12
        self.q_layer = nn.Linear(dim_in, dim_tmp)
        self.k_layer = nn.Linear(dim_in, dim_tmp)
        self.v_layer = nn.Linear(dim_in, dim_tmp)
        self.m_layer = nn.Sequential(
            nn.Linear(edge_in+1, dim_tmp),
            nn.Dropout(p=drop_rate),
            nn.LeakyReLU(), 
            nn.Linear(dim_tmp, dim_tmp)
            )
        self.m2f_layer = nn.Sequential(
            nn.Linear(dim_tmp, dim_tmp),
            nn.Dropout(p=drop_rate))
        self.e_layer = nn.Sequential(
            nn.Linear(dim_tmp, edge_out),
            nn.Dropout(p=drop_rate))
        self.gate_layer = nn.Sequential(
            nn.Linear(3*dim_tmp, dim_tmp),
            nn.Dropout(p=drop_rate))
        self.layer_norm_1 = GraphNorm(dim_tmp)
        self.layer_norm_2 = GraphNorm(dim_tmp)
        self.fin_layer = nn.Sequential(
            nn.Linear(dim_tmp, dim_tmp),
            nn.Dropout(p=drop_rate),
            nn.LeakyReLU(),
            nn.Linear(dim_tmp, dim_tmp)
            )
        self.update_layer = coords_update(dim_dh=self.dh, num_head=num_head, drop_rate=drop_rate)
    
    def forward(self, node_s, edge_s, edge_index, total_pos, pro_nodes, batch, update_pos=True):
        q_ = self.q_layer(node_s)
        k_ = self.k_layer(node_s)
        v_ = self.v_layer(node_s)
        # message passing
        m_ij = torch.cat([edge_s, 
        torch.pairwise_distance(total_pos[edge_index[0]], total_pos[edge_index[1]]).unsqueeze(dim=-1)*0.1], dim=-1)
        m_ij = self.m_layer(m_ij)
        k_ij = k_[edge_index[1]] * m_ij
        a_ij = ((q_[edge_index[0]] * k_ij)/math.sqrt(self.dh)).view((-1, self.num_head, self.dh))
        w_ij = softmax(torch.norm(a_ij, p=1, dim=2), index=edge_index[0]).unsqueeze(dim=-1)
        # update node and edge embeddings 
        node_s_new = self.m2f_layer(scatter(w_ij*v_[edge_index[1]].view((-1, self.num_head, self.dh)), index=edge_index[0], reduce='sum', dim=0).view((-1, self.dim_tmp)))
        edge_s_new = self.e_layer(a_ij.view((-1, self.dim_tmp)))
        g = torch.sigmoid(self.gate_layer(torch.cat([node_s_new, node_s, node_s_new-node_s], dim=-1)))
        node_s_new = self.layer_norm_1(g*node_s_new+node_s, batch)
        node_s_new = self.layer_norm_2(g*self.fin_layer(node_s_new)+node_s_new, batch)
        # update coords
        if update_pos:
            total_pos = self.update_layer(a_ij, total_pos, edge_index, pro_nodes)
        return node_s_new, edge_s_new, edge_index, total_pos


class coords_update(nn.Module):
    def __init__(self, dim_dh, num_head, drop_rate=0.15):
        super().__init__()
        self.num_head = num_head
        self.attention2deltax = nn.Sequential(
            nn.Linear(dim_dh, dim_dh//2),
            nn.Dropout(p=drop_rate),
            nn.LeakyReLU(),
            nn.Linear(dim_dh//2, 1)
        )
        self.weighted_head_layer = nn.Linear(num_head, 1, bias=False)

    def forward(self, a_ij, pos, edge_index, pro_nodes):
        edge_index_mask = edge_index[0] >= pro_nodes
        i, j = edge_index[:, edge_index_mask]
        delta_x = pos[i] - pos[j] 
        delta_x = delta_x/(torch.norm(delta_x, p=2, dim=-1).unsqueeze(dim=-1) + 1e-6 )
        delta_x = delta_x*self.weighted_head_layer(self.attention2deltax(a_ij[edge_index_mask]).squeeze(dim=2))
        delta_x = scatter(delta_x, index=i, reduce='sum', dim=0)
        pos += delta_x
        return pos



