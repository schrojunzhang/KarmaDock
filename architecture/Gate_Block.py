#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   Gate Block
@Time    :   2022/10/13 10:35:49
@Author  :   Xujun Zhang
@Version :   1.0
@Contact :   
@License :   
@Desc    :   None
'''
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.utils import softmax
from torch_scatter import scatter
from torch_geometric.nn import GraphNorm

# here put the import lib
class Gate_Block(nn.Module):
    def __init__(self, dim_tmp, drop_rate=0.15):
        super().__init__()
        self.gate_layer = nn.Sequential(
            nn.Linear(3*dim_tmp, dim_tmp),
            nn.Dropout(p=drop_rate))
        self.norm = GraphNorm(dim_tmp)
    
    def forward(self, f1, f2):
        g = torch.sigmoid(self.gate_layer(torch.cat([f2, f1, f2-f1], dim=-1)))
        f2 = self.norm(g*f2+f1)
        return f2