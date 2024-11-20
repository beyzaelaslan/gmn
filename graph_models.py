# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MetaLayer
from torch_geometric.nn.pool import max_pool_x, avg_pool_x, global_max_pool, global_mean_pool
from torch_scatter import scatter

class EdgeModel(nn.Module):
    def __init__(self, in_dim, out_dim, activation=True):
        super().__init__()
        if activation:
            self.edge_mlp = nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU())
        else:
            self.edge_mlp = nn.Sequential(nn.Linear(in_dim, out_dim))

    def forward(self, src, dest, edge_attr, u, batch):
        print("EdgeModel forward start")
        # **IMPORTANT: YOU ARE NOT ALLOWED TO USE FOR LOOPS!**
        # src, dest: [E, F_x], where E is the number of edges. src is the source node features and dest is the destination node features of each edge.
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs.
        # batch: only here it will have shape [E] with max entry B - 1, because here it indicates the graph index for each edge.

        '''
        Add your code below
        '''
        u_expanded = u[batch]

        # Concatenate all features for each edge
        combined_features = torch.cat([dest, src, edge_attr, u_expanded], dim=1)

        # Apply MLP to the concatenated features
        updated_edge_attr = self.edge_mlp(combined_features)

        return updated_edge_attr

class NodeModel(nn.Module):
    def __init__(self, in_dim_mlp1, in_dim_mlp2, out_dim, activation=True, reduce='sum'):
        super().__init__()
        self.reduce = reduce
        if activation:
            self.node_mlp_1 = nn.Sequential(nn.Linear(in_dim_mlp1, out_dim), nn.ReLU())
            self.node_mlp_2 = nn.Sequential(nn.Linear(in_dim_mlp2, out_dim), nn.ReLU())
        else:
            self.node_mlp_1 = nn.Sequential(nn.Linear(in_dim_mlp1, out_dim))
            self.node_mlp_2 = nn.Sequential(nn.Linear(in_dim_mlp2, out_dim))

    def forward(self, x, edge_index, edge_attr, u, batch):
        print("nodeModel forward start")

        # **IMPORTANT: YOU ARE NOT ALLOWED TO USE FOR LOOPS!**
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.

        '''
        Add your code below
        '''
        # Get the features of the source and destination nodes for each edge
        source_features = x[edge_index[0]]
        dest_features = x[edge_index[1]]

        u_edge = u[batch[edge_index[1]]]

        # Concatenate source and destination node features with edge attributes and global features
        edge_input = torch.cat([dest_features, source_features, edge_attr, u_edge], dim=1)

        # Apply the first MLP
        edge_messages = self.node_mlp_1(edge_input)

        # Aggregate the messages at destination nodes
        aggregated_messages = scatter(edge_messages, edge_index[1], dim=0, reduce=self.reduce, dim_size=x.size(0))

        # Align global features
        u_expanded = u[batch]

        # Concatenate the original node features, aggregated messages, and the global features
        node_input = torch.cat([x, aggregated_messages, u_expanded], dim=1)

        # Apply the second MLP
        updated_node_features = self.node_mlp_2(node_input)

        return updated_node_features

class GlobalModel(nn.Module):
    def __init__(self, in_dim, out_dim, activation=True, reduce='sum'):
        super().__init__()
        if activation:
            self.global_mlp = nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU())
        else:
            self.global_mlp = nn.Sequential(nn.Linear(in_dim, out_dim))
        self.reduce = reduce

    def forward(self, x, edge_index, edge_attr, u, batch):
        #**IMPORTANT: YOU ARE NOT ALLOWED TO USE FOR LOOPS!**
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.

        '''
        Add your code below
        '''
        # Aggregate node and edge features
        node_aggregation = scatter(x, batch, dim=0, reduce=self.reduce)
        edge_batch = batch[edge_index[1]]
        edge_aggregation = scatter(edge_attr, edge_batch, dim=0, reduce=self.reduce)

        # Concatenate aggregated node, edge features, and the global feature
        global_input = torch.cat([node_aggregation, edge_aggregation, u], dim=1)

        # Apply the MLP
        updated_global = self.global_mlp(global_input)
        return updated_global
    
class MPNN(nn.Module):

    def __init__(self, node_in_dim, edge_in_dim, global_in_dim, hidden_dim, node_out_dim, edge_out_dim, global_out_dim, num_layers,
                use_bn=True, dropout=0.1, reduce='sum'):
        super().__init__()
        self.convs = nn.ModuleList()
        self.node_norms = nn.ModuleList()
        self.edge_norms = nn.ModuleList()
        self.global_norms = nn.ModuleList()
        self.use_bn = use_bn
        self.dropout = dropout
        self.reduce = reduce

        assert num_layers >= 2

        '''
        Instantiate the first layer models with correct parameters below
        '''

        # First layer with initial dimensions
        edge_model = EdgeModel(in_dim=2*node_in_dim + edge_in_dim + global_in_dim, out_dim=hidden_dim)
        node_model = NodeModel(in_dim_mlp1= 2*node_in_dim  + hidden_dim + global_in_dim, in_dim_mlp2=(node_in_dim+hidden_dim+global_in_dim), out_dim=hidden_dim)
        global_model = GlobalModel(in_dim=hidden_dim + hidden_dim + global_in_dim, out_dim=hidden_dim)
        self.convs.append(MetaLayer(edge_model=edge_model, node_model=node_model, global_model=global_model))
        self.node_norms.append(nn.BatchNorm1d(hidden_dim))
        self.edge_norms.append(nn.BatchNorm1d(hidden_dim))
        self.global_norms.append(nn.BatchNorm1d(hidden_dim))

        # Intermediate layers with consistent dimensions
        for _ in range(num_layers - 2):
            '''
            Add your code below
            '''
            # add batch norm after each MetaLayer

            edge_model = EdgeModel(in_dim=2*hidden_dim + hidden_dim + hidden_dim, out_dim=hidden_dim)
            node_model = NodeModel(in_dim_mlp1=2*hidden_dim + hidden_dim + hidden_dim, in_dim_mlp2=(hidden_dim+hidden_dim+hidden_dim), out_dim=hidden_dim)
            global_model = GlobalModel(in_dim=hidden_dim + hidden_dim + hidden_dim, out_dim=hidden_dim)

            self.convs.append(MetaLayer(edge_model=edge_model, node_model=node_model, global_model=global_model))
            self.node_norms.append(nn.BatchNorm1d(hidden_dim))
            self.edge_norms.append(nn.BatchNorm1d(hidden_dim))
            self.global_norms.append(nn.BatchNorm1d(hidden_dim))


        '''
        Add your code below
        '''
        # last MetaLayer without batch norm and without using activation functions

        edge_model = EdgeModel(in_dim=2*hidden_dim + hidden_dim + hidden_dim, out_dim=edge_out_dim, activation=False)
        node_model = NodeModel(in_dim_mlp1=2*hidden_dim + edge_out_dim + hidden_dim, in_dim_mlp2=hidden_dim + edge_out_dim + hidden_dim, out_dim=node_out_dim, activation=False)
        global_model = GlobalModel(in_dim=node_out_dim + edge_out_dim + hidden_dim, out_dim=global_out_dim, activation=False)

        self.convs.append(MetaLayer(edge_model=edge_model, node_model=node_model, global_model=global_model))


    def forward(self, x, edge_index, edge_attr, u, batch, *args):
      for i, conv in enumerate(self.convs):
        '''
        Add your code below
        '''
        x, edge_attr, u = conv(x, edge_index, edge_attr, u, batch)

        if i != len(self.convs) - 1 and self.use_bn:
            '''
            Add your code below this line, but before the dropout
            '''
            # Apply batch normalization to the updated node features
            x = self.node_norms[i](x)
            # Apply batch normalization to the updated edge features
            edge_attr = self.edge_norms[i](edge_attr)
            # Apply batch normalization to the updated global features
            u = self.global_norms[i](u)

            x = F.dropout(x, p=self.dropout, training=self.training)
            edge_attr = F.dropout(edge_attr, p=self.dropout, training=self.training)
            u = F.dropout(u, p=self.dropout, training=self.training)

      return x, edge_attr, u
