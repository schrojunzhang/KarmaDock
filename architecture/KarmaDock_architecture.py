#!usr/bin/env python3
# -*- coding:utf-8 -*-
# @time : 2022/8/8 16:51
# @author : Xujun Zhang

import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from architecture.GVP_Block import GVP_embedding
from architecture.GraphTransformer_Block import GraghTransformer
from architecture.MDN_Block import MDN_Block
from architecture.EGNN_Block import EGNN
from architecture.Gate_Block import Gate_Block
from torch_scatter import scatter_mean, scatter
from torch_geometric.nn import GraphNorm


class KarmaDock(nn.Module):
    def __init__(self):
        super(KarmaDock, self).__init__()
        # encoders
        self.lig_encoder = GraghTransformer(
            in_channels=89, 
            edge_features=20, 
            num_hidden_channels=128,
            activ_fn=torch.nn.SiLU(),
            transformer_residual=True,
            num_attention_heads=4,
            norm_to_apply='batch',
            dropout_rate=0.15,
            num_layers=6
        )
        self.pro_encoder = GVP_embedding(
            (9, 3), (128, 16), (102, 1), (32, 1), seq_in=True) 
        self.gn = GraphNorm(128)
        # pose prediction
        self.egnn_layers = nn.ModuleList( 
            [EGNN(dim_in=128, dim_tmp=128, edge_in=128, edge_out=128, num_head=4, drop_rate=0.15) for i in range(8)]
        )
        self.edge_init_layer = nn.Linear(6, 128)
        self.node_gate_layer = Gate_Block(dim_tmp=128, 
                                          drop_rate=0.15
                                          )
        self.edge_gate_layer = Gate_Block(dim_tmp=128, 
                                          drop_rate=0.15
                                          )
        # scoring 
        self.mdn_layer = MDN_Block(hidden_dim=128, 
                                         n_gaussians=10, 
                                        dropout_rate=0.10, 
                                        dist_threhold=7.)


    def cal_rmsd(self, pos_ture, pos_pred, batch, if_r=True):
        if if_r:
            return scatter_mean(((pos_pred - pos_ture)**2).sum(dim=-1), batch).sqrt()
        else:
            return scatter_mean(((pos_pred - pos_ture)**2).sum(dim=-1), batch)

    def cal_rmsd_sym(self, pos_ture, pos_pred, batch_sym_index, batch, if_r=True):
        sym_index_len_size = np.asarray([sym_index.shape for sym_index in batch_sym_index])
        sym_max = sym_index_len_size.max(axis=0)[0]
        batch_sym_index = [np.concatenate([batch_sym_index[idx], np.arange(i[1]).reshape((1, -1)).repeat(sym_max-i[0], axis=0)], axis=0)+sym_index_len_size[:idx][:,1].sum() if i[0]<sym_max else batch_sym_index[idx] +sym_index_len_size[:idx][:,1].sum() for idx, i in enumerate(sym_index_len_size) ]
        batch_sym_index = np.concatenate(batch_sym_index, axis=1)
        return torch.stack([
            self.cal_rmsd(pos_ture, pos_pred[sym_index, :], batch, if_r)
            for sym_index in batch_sym_index], dim=0).min(dim=0).values
    
    def forward(self, data, device, pos_r, recycle_num=3):
        # mdn aux labels
        atom_types_label = torch.argmax(data['ligand'].node_s[:,:18], dim=1, keepdim=False)
        bond_types_label = torch.argmax(data['ligand', 'l2l', 'ligand'].edge_s[data['ligand'].cov_edge_mask][:, :5], dim=1, keepdim=False)
        # encoder 
        pro_node_s = self.pro_encoder((data['protein']['node_s'], data['protein']['node_v']),
                                                      data[(
                                                          "protein", "p2p", "protein")]["edge_index"],
                                                      (data[("protein", "p2p", "protein")]["edge_s"],
                                                       data[("protein", "p2p", "protein")]["edge_v"]),
                                                      data['protein'].seq)
        lig_node_s = self.lig_encoder(data['ligand'].node_s.to(torch.float32), data['ligand', 'l2l', 'ligand'].edge_s[data['ligand'].cov_edge_mask].to(torch.float32), data['ligand', 'l2l', 'ligand'].edge_index[:,data['ligand'].cov_edge_mask])
        # graph norm through a interaction graph
        pro_nodes = data['protein'].num_nodes
        node_s = self.gn(torch.cat([pro_node_s, lig_node_s], dim=0), torch.cat([data['protein'].batch, data['ligand'].batch], dim=-1))
        data['protein'].node_s, data['ligand'].node_s = node_s[:pro_nodes], node_s[pro_nodes:]
        # build interaction graph
        pro_nodes = data['protein'].num_nodes
        batch = torch.cat([data['protein'].batch, data['ligand'].batch], dim=-1)
        u = torch.cat([
            data[("protein", "p2p", "protein")]["edge_index"][0], 
            data[('ligand', 'l2l', 'ligand')]["edge_index"][0]+pro_nodes, 
            data[('protein', 'p2l', 'ligand')]["edge_index"][0], data[('protein', 'p2l', 'ligand')]["edge_index"][1]+pro_nodes], dim=-1)
        v = torch.cat([
            data[("protein", "p2p", "protein")]["edge_index"][1], 
            data[('ligand', 'l2l', 'ligand')]["edge_index"][1]+pro_nodes, 
            data[('protein', 'p2l', 'ligand')]["edge_index"][1]+pro_nodes, data[('protein', 'p2l', 'ligand')]["edge_index"][0]], dim=-1)
        edge_index = torch.stack([u, v], dim=0)
        pki_true = torch.zeros(1, device=device, dtype=torch.float)
        node_s = torch.cat([data['protein'].node_s, data['ligand'].node_s], dim=0)
        edge_s = torch.zeros((data[('protein', 'p2l', 'ligand')]["edge_index"][0].size(0)*2, 6), device=node_s.device)
        edge_s[:, -1] = -1
        edge_s = torch.cat([data[("protein", "p2p", "protein")].full_edge_s, data['ligand', 'l2l', 'ligand'].full_edge_s, edge_s], dim=0)
        pos = torch.cat([data['protein'].xyz, data['ligand'].pos], dim=0)
        # EGNN
        if pos_r:
            edge_s = self.edge_init_layer(edge_s)
            rmsd_losss = torch.tensor([], device=device)
            # 3 recycle 
            for re_idx in range(recycle_num):
                # 8 egnn layer
                for layer in self.egnn_layers:
                    node_s, edge_s, edge_index, pos = layer(node_s, edge_s, edge_index, pos, pro_nodes, batch, update_pos=True)
                    rmsd_losss = torch.cat([rmsd_losss, self.cal_rmsd(pos_ture=data['ligand'].xyz, pos_pred=pos[pro_nodes:], batch=data['ligand'].batch, if_r=True).view((-1, 1))], dim=1)
                # res-connection during each recycling
                node_s = self.node_gate_layer(torch.cat([data['protein'].node_s, data['ligand'].node_s], dim=0), node_s)
                edge_s = self.edge_gate_layer(
                            self.edge_init_layer(torch.cat([
                                data[("protein", "p2p", "protein")].full_edge_s,
                                data['ligand', 'l2l', 'ligand'].full_edge_s,
                                        torch.cat([torch.zeros((data[('protein', 'p2l', 'ligand')]["edge_index"][0].size(0)*2, 5), device=node_s.device),
                                        -torch.ones((data[('protein', 'p2l', 'ligand')]["edge_index"][0].size(0)*2, 1), device=node_s.device),
                                        ], dim=1)], dim=0)), 
                            edge_s)   
            # loss     
            count_idx = random.choice(range(recycle_num))
            rmsd_losss = (rmsd_losss[:, 8*count_idx:8*(count_idx+1)].mean(dim=1) + rmsd_losss[:, -1]).mean()
        else:
            rmsd_losss = torch.zeros(1, device=device, dtype=torch.float)
            frag_losss = torch.zeros(1, device=device, dtype=torch.float)
        # mdn block
        aux_r = 0.001
        lig_pos_ = data['ligand'].xyz
        pi, sigma, mu, dist, _, atom_types, bond_types = self.mdn_layer(lig_s=lig_node_s, lig_pos=lig_pos_, lig_batch=data['ligand'].batch,
                                                               pro_s=pro_node_s, pro_pos=data['protein'].xyz_full, pro_batch=data['protein'].batch,
                                                               edge_index=data['ligand', 'l2l', 'ligand'].edge_index[:, data['ligand'].cov_edge_mask])
        try:
            mdn_loss_true = self.mdn_layer.mdn_loss_fn(pi, sigma, mu, dist)
            mdn_loss_true = mdn_loss_true[torch.where(dist <= self.mdn_layer.dist_threhold)[0]].mean().float() 
            + aux_r*F.cross_entropy(atom_types, atom_types_label) 
            + aux_r*F.cross_entropy(bond_types, bond_types_label)
        except:
            mdn_loss_true = None
    
        return rmsd_losss, mdn_loss_true
    
    def encoding(self, data):
        '''
        get ligand & protein embeddings
        '''
        # get embedding
        pro_node_s = self.pro_encoder((data['protein']['node_s'], data['protein']['node_v']),
                                                      data[(
                                                          "protein", "p2p", "protein")]["edge_index"],
                                                      (data[("protein", "p2p", "protein")]["edge_s"],
                                                       data[("protein", "p2p", "protein")]["edge_v"]),
                                                      data['protein'].seq)
        lig_node_s = self.lig_encoder(data['ligand'].node_s.to(torch.float32), data['ligand', 'l2l', 'ligand'].edge_s[data['ligand'].cov_edge_mask].to(torch.float32), data['ligand', 'l2l', 'ligand'].edge_index[:,data['ligand'].cov_edge_mask])
        return pro_node_s, lig_node_s
    
    def scoring(self, lig_s, lig_pos, pro_s, data, dist_threhold, batch_size):
        '''
        scoring the protein-ligand binding strength
        '''
        pi, sigma, mu, dist, c_batch, _, _ = self.mdn_layer(lig_s=lig_s, lig_pos=lig_pos, lig_batch=data['ligand'].batch,
                                                               pro_s=pro_s, pro_pos=data['protein'].xyz_full, pro_batch=data['protein'].batch,
                                                               edge_index=data['ligand', 'l2l', 'ligand'].edge_index[:, data['ligand'].cov_edge_mask])
        mdn_score = self.mdn_layer.calculate_probablity(pi, sigma, mu, dist)
        mdn_score[torch.where(dist > dist_threhold)[0]] = 0.
        mdn_score = scatter(mdn_score, index=c_batch, dim=0, reduce='sum', dim_size=batch_size).float()
        return mdn_score
    
    def docking(self, pro_node_s, lig_node_s, data, recycle_num=3):
        '''
        generate protein-ligand binding conformations 
        '''
        # graph norm through interaction graph
        pro_nodes = data['protein'].num_nodes
        node_s = self.gn(torch.cat([pro_node_s, lig_node_s], dim=0), torch.cat([data['protein'].batch, data['ligand'].batch], dim=-1))
        data['protein'].node_s, data['ligand'].node_s = node_s[:pro_nodes], node_s[pro_nodes:]
        # build interaction graph
        pro_nodes = data['protein'].num_nodes
        batch = torch.cat([data['protein'].batch, data['ligand'].batch], dim=-1)
        u = torch.cat([data[("protein", "p2p", "protein")]["edge_index"][0], data[('ligand', 'l2l', 'ligand')]["edge_index"][0]+pro_nodes, data[('protein', 'p2l', 'ligand')]["edge_index"][0], data[('protein', 'p2l', 'ligand')]["edge_index"][1]+pro_nodes], dim=-1)
        v = torch.cat([data[("protein", "p2p", "protein")]["edge_index"][1], data[('ligand', 'l2l', 'ligand')]["edge_index"][1]+pro_nodes, data[('protein', 'p2l', 'ligand')]["edge_index"][1]+pro_nodes, data[('protein', 'p2l', 'ligand')]["edge_index"][0]], dim=-1)
        edge_index = torch.stack([u, v], dim=0)
        node_s = torch.cat([data['protein'].node_s, data['ligand'].node_s], dim=0)
        edge_s = torch.zeros((data[('protein', 'p2l', 'ligand')]["edge_index"][0].size(0)*2, 6), device=node_s.device)
        edge_s[:, -1] = -1
        edge_s = torch.cat([data[("protein", "p2p", "protein")].full_edge_s, data['ligand', 'l2l', 'ligand'].full_edge_s, edge_s], dim=0)
        pos = torch.cat([data['protein'].xyz, data['ligand'].pos], dim=0)
        # EGNN
        edge_s = self.edge_init_layer(edge_s)
        for re_idx in range(recycle_num):
            for layer in self.egnn_layers:
                node_s, edge_s, edge_index, pos = layer(node_s, edge_s, edge_index, pos, pro_nodes, batch, update_pos=True)
            node_s = self.node_gate_layer(torch.cat([data['protein'].node_s, data['ligand'].node_s], dim=0), node_s)
            edge_s = self.edge_gate_layer(
                            self.edge_init_layer(torch.cat([
                                data[("protein", "p2p", "protein")].full_edge_s,
                                data['ligand', 'l2l', 'ligand'].full_edge_s,
                                        torch.cat([torch.zeros((data[('protein', 'p2l', 'ligand')]["edge_index"][0].size(0)*2, 5), device=node_s.device),
                                        -torch.ones((data[('protein', 'p2l', 'ligand')]["edge_index"][0].size(0)*2, 1), device=node_s.device),
                                        ], dim=1)], dim=0)), 
                            edge_s)  
        return pos[pro_nodes:], data['ligand'].xyz, data['ligand'].batch
    
    def ligand_docking(self, data, docking=False, scoring=False, recycle_num=3, dist_threhold=5):
        '''
        generating protein-ligand binding conformations and  predicting their binding strength
        '''
        device = data['protein'].node_s.device
        batch_size = data['protein'].batch.max()+1
        # encoder
        pro_node_s, lig_node_s = self.encoding(data)
        # docking
        if docking:
            lig_pos, _, _ = self.docking(pro_node_s, lig_node_s, data, recycle_num)
        else:
            lig_pos = data['ligand'].xyz
        # scoring
        if scoring:
            mdn_score = self.scoring(lig_s=lig_node_s, lig_pos=lig_pos, pro_s=pro_node_s, data=data,
                                                               dist_threhold=dist_threhold, batch_size=batch_size)
        else:
            mdn_score = torch.zeros(len(data), device=device, dtype=torch.float)
        return lig_pos, mdn_score
    