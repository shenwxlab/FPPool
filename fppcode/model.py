# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Sequential, Linear, ModuleList, ReLU, Dropout, Conv1d

from torch_geometric.nn import MessagePassing, JumpingKnowledge
from torch_geometric.nn import NNConv, GATv2Conv, PNAConv, SAGEConv, GINEConv, MLP 
from torch_geometric.nn import global_mean_pool, global_max_pool, Set2Set, GlobalAttention
from torch_geometric.nn import (GINConv, GCNConv, GATConv, SAGEConv,
                                DenseGINConv, DenseGCNConv,
                                AttentiveFP, MLP,
                                TopKPooling, SAGPooling, dense_diff_pool, dense_mincut_pool, EdgePooling, ASAPooling,
                                GraphMultisetTransformer, global_add_pool,
                                BatchNorm)
from torch_geometric.utils import degree, softmax, to_dense_batch, to_dense_adj
from copy import deepcopy 
from .pooling import local_substructure_pool, SubstructurePool
from torch_geometric.nn import pool, aggr
import json
import re
def fix_reproducibility(seed=42):
    import os, random
    import numpy as np
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed),
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    

def clean_filename(filename):
    invalid_chars = r'<>:"/\\|?*[]='
    cleaned_filename = filename
    for char in invalid_chars:
        cleaned_filename = cleaned_filename.replace(char, "_")
    cleaned_filename = re.sub(r'[\x00-\x1F\x7F]', '_', cleaned_filename)
    
    return cleaned_filename
    
class GNN_Base(torch.nn.Module):

    r"""An base class for GNN
    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each out sample.
        edge_dim (int): Edge feature dimensionality.
        fp_dim (int): Node substructure fingerprint dimensionality, 881 for PubchemFP, 166 for MACCSFP, ...

        convs_layers: Message passing layers. (default: :[128, 64, 1])
        pool_layer: (torch_geometric.nn.Module, optional): the pooling-layer. (default: :obj:  SubstructurePool(reduce='mean'))
                    if the pooling layer is the global pooling (e. g., torch_geometric.nn.global_mean_pool), 
                    the last layer of convs_layers should be laaaarger, such as [64, 128, 512]
        dense_layers: Fully-connected layers. (default: :[512, 128, 32])
        batch_norms (torch.nn.Module, optional, say torch.nn.BatchNorm1d): The normalization operator to use. (default: :obj:`None`)
        dropout_p (float, optional): Dropout probability. (default: :obj:`0.1`) of ACNet, different from dropout in GATConv layer
        **kwargs (optional): Additional arguments of the underlying:class:`torch_geometric.nn.conv.MessagePassing` layers.
    """
    
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 edge_dim,
                 fp_dim = None,  
                 convs_layers = [128, 64, 1],
                 pooling_layer = SubstructurePool(reduce='sum'),
                 dense_layers = [512, 128, 32],
                 batch_norms = torch.nn.BatchNorm1d,
                 dropout_p = 0.0,
                 **kwargs,
                ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        self.fp_dim = fp_dim
        self.max_nodes = 150
        
        self.convs_layers = convs_layers
        self.dropout_p = dropout_p
 
        self.pooling_layer = pooling_layer
        self.dense_layers = dense_layers

        ### CONVS STACKs
        _convs_layers = [in_channels]
        _convs_layers.extend(convs_layers)
        self._convs_layers = _convs_layers
        self.convs = ModuleList()
        for i in range(len(_convs_layers)-1):
            convs = self.init_conv(_convs_layers[i], _convs_layers[i+1], edge_dim, **kwargs)
            self.convs.append(convs)
        ### NORM STACKs
        self.batch_norms = None
        if batch_norms is not None:
            self.batch_norms = ModuleList()
            for i in range(len(_convs_layers)-1):
                self.batch_norms.append(deepcopy(batch_norms(_convs_layers[i+1])))

        self.gate1 = Linear(_convs_layers[-1], 1)
        self.gate2 = Conv1d(_convs_layers[-1], 1, kernel_size=1)

        if pooling_layer == 'diffpool' or 'mincutpool' or 'a':
            self.pool_conv_nets = nn.ModuleList()
            self.score_nets = nn.ModuleList()
            self.ratio = 0.5
            pool_in_channels = _convs_layers[-1]
            #num_nodes = math.ceil(self.ratio * self.max_nodes)
            num_nodes = self.max_nodes
            for i in range(2):
                num_nodes = math.ceil(self.ratio * num_nodes)
                pool_net = DenseGINConv(Sequential(Linear(pool_in_channels, pool_in_channels), ReLU()))
                score_net = DenseGINConv(Sequential(Linear(pool_in_channels, num_nodes), ReLU()))
                self.pool_conv_nets.append(pool_net)
                self.score_nets.append(score_net)
        if pooling_layer == 'topkpool' :
            self.pool_conv_nets = nn.ModuleList()
            self.pool_nets = nn.ModuleList()
            pool_in_channels = _convs_layers[-1]
            for i in range(2):
                pool_conv_net = GINConv(Sequential(Linear(pool_in_channels, pool_in_channels)))
                pool_net = TopKPooling(in_channels=pool_in_channels, ratio=self.ratio)
                self.pool_conv_nets.append(pool_conv_net)
                self.pool_nets.append(pool_net)
        if pooling_layer == 'sagpool':
            self.pool_conv_nets = nn.ModuleList()
            self.pool_nets = nn.ModuleList()
            pool_in_channels = _convs_layers[-1]
            for i in range(2):
                pool_conv_net = GINConv(Sequential(Linear(pool_in_channels, pool_in_channels)))
                pool_net = SAGPooling(in_channels=pool_in_channels, ratio=self.ratio)
                self.pool_conv_nets.append(pool_conv_net)
                self.pool_nets.append(pool_net)

        if self.pooling_layer=='set2set':
            _dense_layers = [_convs_layers[-1]*2]
        else:
            _dense_layers = [_convs_layers[-1]]
        _dense_layers.extend(dense_layers)
        self._dense_layers = _dense_layers
        
        # LINEAR PRED LAYER
        self.lins = ModuleList()
        for i in range(len(_dense_layers)-1):
            lin = Linear(_dense_layers[i], _dense_layers[i+1])
            self.lins.append(lin)
        # OUTPUT PRED LAYER
        last_hidden = _dense_layers[-1]
        self.out = Linear(last_hidden, out_channels)
        
        if batch_norms is not None:
            batch_norms_name = batch_norms.__name__
        else:
            batch_norms_name = None

        model_args = {'in_channels':self.in_channels, 
                'out_channels':self.out_channels,
                'edge_dim':self.edge_dim, 
                'fp_dim':self.fp_dim,
                'convs_layers': self.convs_layers, 
                'dropout_p':self.dropout_p, 
                'batch_norms':batch_norms_name,
                #'pooling_layer':self.pooling_layer.__name__,
                'pooling_layer':self.pooling_layer,
                'dense_layers':self.dense_layers,
               }
        
        for k, v in kwargs.items():
            if type(v) == torch.Tensor:
                v1 = v.numpy().tolist()
                model_args[k] = v1
            else:
                model_args[k] = v
            setattr(self, k, v)   
            
        self.model_args = model_args


    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()
        for norm in self.batch_norms or []:
            norm.reset_parameters()
        if hasattr(self, 'out'):
            self.out.reset_parameters()          
        if hasattr(self, 'gate1'):
            self.out.reset_parameters()  
        if hasattr(self, 'gate2'):
            self.out.reset_parameters()  
            
        
    def init_conv(self, in_channels, out_channels,  **kwargs):
        raise NotImplementedError
        

    def forward(self, x, edge_index, edge_attr, batch, fp, fp_length, *args, **kwargs):
        '''
        data.x, data.edge_index, data.edge_attr, data.batch, data.fp,...
        '''
        pool_loss = 0
        contributes = None
        mask = None

        for i, convs in enumerate(self.convs):
            x = convs(x, edge_index, edge_attr)# *args, **kwargs)        
            x = F.relu(x, inplace=True)
            if self.batch_norms is not None:
                x = self.batch_norms[i](x)

        if not isinstance(self.pooling_layer, str):
            embed, contributes, mask = self.pooling_layer(x, batch, fp)
            y = embed.squeeze(dim=-1)
            embed = embed.squeeze(dim=-1)

        # FLAT POOLING METHODS
        elif self.pooling_layer=='sum':
            y = pool.global_add_pool(x, batch)
            embed = y
        elif self.pooling_layer=='max':
            y = pool.global_max_pool(x, batch)
            embed = y
        elif self.pooling_layer=='mean':
            y = pool.global_mean_pool(x, batch)
            embed = y
        elif self.pooling_layer=='set2set':
            pool_in_channels = self._convs_layers[-1]
            pool_net = aggr.Set2Set(in_channels=pool_in_channels, processing_steps=2).to(x.device)
            x = pool_net(x, batch)
            y = x
            embed = x

        ### SPARSE POOLING METHODS
        elif self.pooling_layer=='sagpool':
            for i in range(2):
                batch_ = batch
                x = self.pool_conv_nets[i](x, edge_index)
                x, edge_index, _, batch, perm, score = self.pool_nets[i](x=x, edge_index=edge_index, batch=batch)
                if i == 0:
                    contributes = (perm, batch_)
                    
            x = pool.global_add_pool(x, batch)
            y = x
            embed = x
        elif self.pooling_layer=='topkpool':
            for i in range(2):
                batch_ = batch
                x, edge_index, _, batch, perm, score = self.pool_nets[i](x=x, edge_index=edge_index, batch=batch)
                if i == 0:
                    contributes = (perm, batch_)
            x = pool.global_add_pool(x, batch)
            y = x
            embed = x
            


        ### DENSE POOLING METHODS
        elif self.pooling_layer=='diffpool':
            pool_loss = 0
            x, mask = to_dense_batch(x, batch, max_num_nodes=self.max_nodes)
            adj = to_dense_adj(edge_index, batch, max_num_nodes=self.max_nodes)
            for i in range(2):
                x = self.pool_conv_nets[i](x, adj, mask)
                s = self.score_nets[i](x, adj, mask)              
                x, adj, l1, l2 = dense_diff_pool(x, adj, s)
                l = 0.5*l1 + 0.5*l2
                pool_loss = pool_loss + l
                mask = None

            y = x.sum(dim = 1)
            embed = x.sum(dim = 1)
        elif self.pooling_layer=='asapool':
            pool_loss = 0
            x, mask = to_dense_batch(x, batch, max_num_nodes=self.max_nodes)
            adj = to_dense_adj(edge_index, batch, max_num_nodes=self.max_nodes)
            for i in range(2):
                x = self.pool_conv_nets[i](x, adj, mask)
                s = self.score_nets[i](x, adj, mask)              
                x, adj, l1, l2 = dense_diff_pool(x, adj, s)
                l = 0.5*l1 + 0.5*l2
                pool_loss = pool_loss + l
                mask = None

            y = x.sum(dim = 1)
            embed = x.sum(dim = 1)
        elif self.pooling_layer=='mincutpool':
            pool_loss = 0
            x, mask = to_dense_batch(x, batch, max_num_nodes=self.max_nodes)
            adj = to_dense_adj(edge_index, batch, max_num_nodes=self.max_nodes)
            for i in range(2):
                x = self.pool_conv_nets[i](x, adj, mask)
                s = self.score_nets[i](x, adj, mask)              
                x, adj, l1, l2 = dense_mincut_pool(x, adj, s)
                l = 0.5*l1 + 0.5*l2
                pool_loss = pool_loss + l
                mask = None

            y = x.sum(dim = 1)
            embed = x.sum(dim = 1)
        
        ### PASS TO LINEAR LAYER
        for lin in self.lins:
            y = F.relu(lin(y), inplace=True)
            y = F.dropout(y, p=self.dropout_p, training=self.training)
        ### PASS TO OUTPUT LAYER
        y = self.out(y)

        return y, embed, contributes, mask, pool_loss
    

class GNN_GCN(GNN_Base):
    r"""The continuous kernel-based convolutional operator from the
    `"Neural Message Passing for Quantum Chemistry"
    <https://arxiv.org/abs/1704.01212>`_ paper, using the
    :class:`~torch_geometric.nn.conv.NNConv` operator for message passing.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each out sample.
        edge_dim (int): Edge feature dimensionality.
        fp_dim (int): Node substructure fingerprint dimensionality, 881 for PubchemFP, 166 for MACCSFP, ...
        convs_layers: Message passing layers. (default: :[128, 64, 1])
        pool_layer: (torch_geometric.nn.Module, optional): the pooling-layer. (default: :obj:  SubstructurePool(reduce='mean'))
                    if the pooling layer is the global pooling (e. g., torch_geometric.nn.global_mean_pool), 
                    the last layer of convs_layers should be laaaarger, such as [64, 128, 512]
        dense_layers: Fully-connected layers. (default: :[512, 128, 32])
        batch_norms (torch.nn.Module, optional, say torch.nn.BatchNorm1d): The normalization operator to use. (default: :obj:`None`)
        dropout_p (float, optional): Dropout probability. (default: :obj:`0.1`) of ACNet, different from dropout in GATConv layer
        **kwargs (optional): Additional arguments of the underlying:class:`torch_geometric.nn.conv.MessagePassing` layers.
    """
    
    # integrate multi-dimensional edge features, GCNConv only accepts 1-d edge features
    def init_conv(self, in_channels, out_channels, edge_dim, **kwargs):
        
        # To map each edge feature to a vector of shape (in_channels * out_channels) as weight to compute messages.
        edge_fuc = Sequential(Linear(edge_dim, in_channels * out_channels))
        return NNConv(in_channels, out_channels, nn = edge_fuc, **kwargs)

    
class GNN_GIN(GNN_Base):

    r"""The modified :class:`GINConv` operator from the `"Strategies for
    Pre-training Graph Neural Networks" <https://arxiv.org/abs/1905.12265>`_
    paper, using the :class:`~torch_geometric.nn.GINEConv` operator for message passing.
    It is able to corporate edge features into the aggregation procedure. 


    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each out sample.
        edge_dim (int): Edge feature dimensionality.
        fp_dim (int): Node substructure fingerprint dimensionality, 881 for PubchemFP, 166 for MACCSFP, ...
        convs_layers: Message passing layers. (default: :[128, 64, 1])
        pool_layer: (torch_geometric.nn.Module, optional): the pooling-layer. (default: :obj:  SubstructurePool(reduce='mean'))
                    if the pooling layer is the global pooling (e. g., torch_geometric.nn.global_mean_pool), 
                    the last layer of convs_layers should be laaaarger, such as [64, 128, 512]
        dense_layers: Fully-connected layers. (default: :[512, 128, 32])
        batch_norms (torch.nn.Module, optional, say torch.nn.BatchNorm1d): The normalization operator to use. (default: :obj:`None`)
        dropout_p (float, optional): Dropout probability. (default: :obj:`0.1`) of ACNet, different from dropout in GATConv layer
        **kwargs (optional): Additional arguments of the underlying:class:`torch_geometric.nn.conv.MessagePassing` layers.
    """

    def init_conv(self, in_channels, out_channels, edge_dim, **kwargs): 
        #A neural network :math:`h_{\mathbf{\Theta}}` that maps node features `x` of shape `[-1, in_channels]` to `[-1, out_channels]`
        node_fuc = Sequential(Linear(in_channels, out_channels))
        return GINEConv(nn = node_fuc, edge_dim = edge_dim, **kwargs)
    
    
    
class GNN_GAT(GNN_Base):
    r"""The Graph Neural Network from `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ or `"How Attentive are Graph Attention
    Networks?" <https://arxiv.org/abs/2105.14491>`_ papers, using the
    :class:`~torch_geometric.nn.GATv2Conv` operator for message passing.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each out sample.
        edge_dim (int): Edge feature dimensionality.
        fp_dim (int): Node substructure fingerprint dimensionality, 881 for PubchemFP, 166 for MACCSFP, ...
        convs_layers: Message passing layers. (default: :[128, 64, 1])
        pool_layer: (torch_geometric.nn.Module, optional): the pooling-layer. (default: :obj:  SubstructurePool(reduce='mean'))
                    if the pooling layer is the global pooling (e. g., torch_geometric.nn.global_mean_pool), 
                    the last layer of convs_layers should be laaaarger, such as [64, 128, 512]
        dense_layers: Fully-connected layers. (default: :[512, 128, 32])
        batch_norms (torch.nn.Module, optional, say torch.nn.BatchNorm1d): The normalization operator to use. (default: :obj:`None`)
        dropout_p (float, optional): Dropout probability. (default: :obj:`0.1`) of ACNet, different from dropout in GATConv layer
        **kwargs (optional): Additional arguments of the underlying:class:`torch_geometric.nn.conv.MessagePassing` layers.
    """
    
    def init_conv(self, in_channels, out_channels, edge_dim, **kwargs): 
        #False concat the head, to average the information
        concat = kwargs.pop('concat', False)
        return GATv2Conv(in_channels, out_channels, edge_dim = edge_dim, concat=concat, **kwargs)

    
    
    
class GNN_PNA(GNN_Base):
    r"""The Graph Neural Network from the `"Principal Neighbourhood Aggregation
    for Graph Nets" <https://arxiv.org/abs/2004.05718>`_ paper, using the
    :class:`~torch_geometric.nn.conv.PNAConv` operator for message passing.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each out sample.
        edge_dim (int): Edge feature dimensionality.
        fp_dim (int): Node substructure fingerprint dimensionality, 881 for PubchemFP, 166 for MACCSFP, ...
        convs_layers: Message passing layers. (default: :[128, 64, 1])
        pool_layer: (torch_geometric.nn.Module, optional): the pooling-layer. (default: :obj:  SubstructurePool(reduce='mean'))
                    if the pooling layer is the global pooling (e. g., torch_geometric.nn.global_mean_pool), 
                    the last layer of convs_layers should be laaaarger, such as [64, 128, 512]
        dense_layers: Fully-connected layers. (default: :[512, 128, 32])
        batch_norms (torch.nn.Module, optional, say torch.nn.BatchNorm1d): The normalization operator to use. (default: :obj:`None`)
        dropout_p (float, optional): Dropout probability. (default: :obj:`0.1`) of ACNet, different from dropout in GATConv layer
        aggregators(list of str): Set of aggregation function identifiers, e.g., ['mean', 'min', 'max', 'sum']
        scalers(list of str): Set of scaling function identifiers, e.g., ['identity', 'amplification', 'attenuation'] 
        deg (Tensor): Histogram of in-degrees of nodes in the training set, used by scalers to normalize, e.g.,  torch.tensor([1, 2, 3]
        **kwargs (optional): Additional arguments of the underlying:class:`torch_geometric.nn.conv.MessagePassing` layers.
    """

    def init_conv(self, in_channels, out_channels, edge_dim, **kwargs): 
        return PNAConv(in_channels, out_channels, edge_dim = edge_dim,  **kwargs)

                

def get_deg(train_dataset):
    # Compute the maximum in-degree in the training data.
    max_degree = -1
    for data in train_dataset:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        max_degree = max(max_degree, int(d.max()))

    # Compute the in-degree histogram tensor
    deg = torch.zeros(max_degree + 1, dtype=torch.long)
    for data in train_dataset:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())
    return deg



__all__ = ['ACANet_GCN', 'ACANet_GIN', 'ACANet_GAT', 'ACANet_PNA']