# -*- coding: utf-8 -*-
import torch
from torch import nn
from torch import Tensor
from torch.nn import ModuleList, Sequential, Linear, Module
from torch_geometric.nn.aggr import Aggregation
from torch_geometric.nn.inits import reset
from torch_geometric.utils import softmax, to_dense_batch
from typing import Optional
from functools import partial

class SubstructurePool:

    def __init__(self, reduce = 'sum'):
        '''
        reduce: sum, mean, max, min
        '''
        self.reduce = reduce
        self.__name__ = 'SubstructurePool'
        
        
    def __call__(self, x, batch, fp):
        
        if self.reduce in ('sum', 'add', 'max'):
            return _local_substructure_pool(x, batch, fp, self.reduce) # more faster method
        else:
            return local_substructure_pool(x, batch, fp, self.reduce)


    
def local_substructure_pool(x, batch, fp, reduce = 'sum'):

    '''
    LSP: Returns batch-wise subgraph-level-output, 3D tensor, (batch_size, fingerprint_dim, in_channel), 
    if in_channel is 1, batch size is 32, fingerprint dim is 1024, then the output shape is : (32, 1024, 1)
    
     Args:
        x (Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times F}`.
        batch (LongTensor): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example.
        fp (Tensor): node atom fingerprint matrix (MxN, N: 1024, M: 312, number of nodes).
            The subgraph info for each sample
        reduce: `sum` (`add`)`max`, `mean` and `min`..
    '''
    ## in_channel = 1
    atom_size, in_channel = x.shape
    sample_size = int(batch.max().item() + 1)
    fp = fp.to(batch.dtype)
    
    ## sub-structure pooling for each channel
    all_channel_pool = [] 
    for i in range(in_channel): # per-channel pooling, in_channel equals 1 in our GNN model
        x1 = x[:, i] #
        all_mol_pool = []
        for s in range(sample_size): # per-mol pooling
            sample_mask = batch == s
            mol_fp = fp[sample_mask]
            mol_x = x1[sample_mask]
            mol_x = broadcast(mol_x, mol_fp, dim=-2)
            mol_pool_out = scatter(mol_x,  mol_fp,  dim=-2, dim_size=2, reduce=reduce)
            offbit, onbit = mol_pool_out #{0 & 1, same as batch 0,1,2,3,4}, we only need to collect 1.        
            all_mol_pool.append(onbit)
        one_channel_pool = torch.stack(all_mol_pool)
        all_channel_pool.append(one_channel_pool)
            
    #3D tensor, (batch_size, fingerprint_dim, in_channel)
    substructure_pool = torch.stack(all_channel_pool, axis=-1) 

    return substructure_pool



def _local_substructure_pool(x, batch, fp, reduce = 'sum'):


    '''
    LSP: Returns batch-wise subgraph-level-output, 3D tensor, (batch_size, fingerprint_dim, in_channel), 
    if in_channel is 1, batch size is 32, fingerprint dim is 1024, then the output shape is : (32, 1024, 1)
    
     Args:
        x (Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times F}`.
        batch (LongTensor): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example.
        fp (Tensor): node atom fingerprint matrix (MxN, N: 1024, M: 312, number of nodes).
            The subgraph info for each sample
        reduce: `sum` (`add`) or `max`.
            Please note: `mean` and `min` are not suitable in this function,
            Rewrite pool if you want the "mean" and "min" operation.
    '''

    num_atom, in_channel = x.shape
    size = int(batch.max().item() + 1)
    fp = fp.to(x.dtype)
    
    ## sub-structure pooling for each channel
    per_channel_pool_res = []
    for i in range(in_channel):
        x1 = x[:, i] #
        x1 = broadcast(x1, fp, dim=-2)
        x1 = torch.mul(x1, fp)
        pool_out = scatter(x1, batch, dim=-2, dim_size=size, reduce=reduce)
        per_channel_pool_res.append(pool_out)

    #3D tensor, (batch_size, fingerprint_dim, in_channel)
    substructure_pool_res = torch.stack(per_channel_pool_res, axis=-1) 

    return substructure_pool_res


class AttentionalAggregation(Aggregation):
    r"""The soft attention aggregation layer from the `"Graph Matching Networks
    for Learning the Similarity of Graph Structured Objects"
    <https://arxiv.org/abs/1904.12787>`_ paper

    .. math::
        \mathbf{r}_i = \sum_{n=1}^{N_i} \mathrm{softmax} \left(
        h_{\mathrm{gate}} ( \mathbf{x}_n ) \right) \cdot
        h_{\mathbf{\Theta}} ( \mathbf{x}_n ),

    where :math:`h_{\mathrm{gate}} \colon \mathbb{R}^F \to
    \mathbb{R}` and :math:`h_{\mathbf{\Theta}}` denote neural networks, *i.e.*
    MLPs.

    Args:
        gate_nn (torch.nn.Module): A neural network :math:`h_{\mathrm{gate}}`
            that computes attention scores by mapping node features :obj:`x` of
            shape :obj:`[-1, in_channels]` to shape :obj:`[-1, 1]` (for
            node-level gating) or :obj:`[1, out_channels]` (for feature-level
            gating), *e.g.*, defined by :class:`torch.nn.Sequential`.
        nn (torch.nn.Module, optional): A neural network
            :math:`h_{\mathbf{\Theta}}` that maps node features :obj:`x` of
            shape :obj:`[-1, in_channels]` to shape :obj:`[-1, out_channels]`
            before combining them with the attention scores, *e.g.*, defined by
            :class:`torch.nn.Sequential`. (default: :obj:`None`)
    """
    def __init__(self, gate_nn: torch.nn.Module,
                 nn: Optional[torch.nn.Module] = None):
        super().__init__()
        self.gate_nn = gate_nn
        self.nn = nn
        self.reset_parameters()
        
    def reset_parameters(self):
        reset(self.gate_nn)
        reset(self.nn)
        
    def forward(self, mask= None,x= None,  index= None,
                ptr= None, dim_size = None,
                dim = -2) -> Tensor:
        '''
        if mask == None:
            mask = torch.ones(len(x)).to(x.device).to(torch.bool)
        '''
        x = x
        if index == None:
            index = torch.zeros(len(x)).to(x.device).to(torch.long)          
        self.assert_two_dimensional_input(x, dim)
        gate = self.gate_nn(x)
        x = self.nn(x) if self.nn is not None else x
        gate = softmax(gate, index=index)
        return self.reduce(gate * x, index=index)
    
    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(gate_nn={self.gate_nn}, '
                f'nn={self.nn})')



class FingerprintPool(Module):
    def __init__(self, in_channel, hidden_channel, fp_length, reduce_inner = 'attn', reduce_inter = 'attn', reduce_global= 'attn', atoms_repr=True):
        '''
        reduce type: attn, sum, mean, max
        '''
        super(FingerprintPool,self).__init__()
        self.__name__ = 'FingerprintPool'
        self.in_channel = in_channel
        self.hidden_channel = hidden_channel
        self.reduce_inner = reduce_inner
        self.reduce_inter = reduce_inter
        self.reduce_global = reduce_global
        self.atoms_repr = atoms_repr
        self.fp_length = fp_length
        # if atoms_repr is True: every single atom can be viewed as a specail cluster
        if atoms_repr:
            fp_num = len(fp_length) + 1
        else:
            fp_num = len(fp_length)
        
        if self.reduce_inner == 'attn':
            self.gate_inner = nn.ModuleList([nn.Linear(in_channel, 1) for _ in range(fp_num)])
        if self.reduce_inter == 'attn':
            self.gate_inter = nn.ModuleList([nn.Linear(in_channel, 1) for _ in range(fp_num)])
        if self.reduce_global == 'attn':
            self.gate_global = nn.Linear(in_channel, 1)

    def fingerprint_pooling(self, x, fp, fp_length, reduce_sub, reduce_global, mask):

        if reduce == 'sum':
            pool_per_fingerprint = torch.stack(list(map(lambda a: x_[a].sum(0), fp_T)), dim=0)
        if reduce == 'mean':
            pool_per_fingerprint = torch.stack(list(map(lambda a: x_[a].mean(0), fp_T)), dim=0)
        if reduce == 'max':
            pool_per_fingerprint = torch.stack(list(map(lambda a: x_[a].max(0)[0], fp_T)), dim=0)
        
        return final_repr
    
    def forward(self, x, batch, fp):
        
        fp_length = self.fp_length
        num_atom, in_channel = x.shape
        size = int(batch.max().item() + 1)

        fp = fp.to(torch.int) #[N, FP]
        fp_length = fp_length.to(torch.int).to(x.device)
        
        if self.atoms_repr:
            atoms_mask = torch.ones((num_atom,1)).to(x.device)
            fp = torch.concat((atoms_mask,fp), dim=-1)
            fp_length = torch.concat((torch.tensor([1]).to(x.device), fp_length))
        
        x_dense, x_dense_mask = to_dense_batch(x, batch) #[batch, Nmax, in_dim]
        fp_dense, fp_dense_mask = to_dense_batch(fp, batch) #[batch, Nmax, allfp_dim]
        fp_mask = fp_dense.permute(0,2,1).unsqueeze(-1).repeat(1,1,1,in_channel) #[batch, allfp_dim, Nmax, in_dim]

        fp_atom_num = fp_dense.permute(0,2,1).sum(dim=-1).unsqueeze(-1) #(batch, allfp_dim, 1)
        x_fp = x_dense.unsqueeze(1)*fp_mask #[batch, allfp_dim, Nmax, in_dim]
        start = 0
        fp_range = []
        for i in fp_length:
            end = start + i
            fp_range.append((start,end))
            start = end
        
        # INNER LEVEL POOLING
        if self.reduce_inner =='sum':
            x_inner = x_fp.sum(-2) # (batch, fp_dim, in_dim)
        elif self.reduce_inner =='mean':
            x_inner = x_fp.sum(-2)/fp_atom_num 
        elif self.reduce_inner == 'max':
            fp_nohit_mask = fp_dense.permute(0,2,1).unsqueeze(-1) < 1e-12 # (batch, fp_dim, Nmax, 1)
            x_fp = x_fp + torch.tensor(float('-inf')) * fp_nohit_mask.to(torch.int)
            x_inner = x_fp.max(-2)
        elif self.reduce_inner == 'attn':
            x_inner_list = []
            contributes_inner_list = []
            for i, gate in enumerate(self.gate_inner):
                x_per_fp = x_fp[:, fp_range[i][0]:fp_range[i][1], :, :] # (batch, fp_i, Nmax, in_dim)
                mask_per_fp = fp_dense.permute(0,2,1).unsqueeze(-1)[:, fp_range[i][0]:fp_range[i][1], :] < 1e-5 # (batch, fp_i_dim, Nmax, 1)  
                score = gate(x_per_fp) # (batch, fp_i_dim, Nmax, 1)
                score_mask_bias = mask_per_fp * (-1e12)
                score = score + score_mask_bias
                inner_score_softmax = torch.softmax(score, dim = -2) # (batch, fp_i, Nmax, 1)
                x_inner = inner_score_softmax * x_per_fp # (batch, fp_i, Nmax, in_dim) 
                x_inner = x_inner.sum(-2) # (batch, fp_i, in_dim)
                x_inner_list.append(x_inner)  # [(batch, fp_1, in_dim), (batch, fp_2, in_dim), ...]
                contributes_inner_list.append(inner_score_softmax)   # [(batch, fp_1, Nmax, 1), (batch, fp_2, Nmax, 1), ...]


        # INTER LEVEL POOLING
        if self.reduce_inter =='sum':
            x_inter_list = []
            for i in x_inner_list:
                x_inter = x_inner.sum(-2) # (batch, in_dim)
                x_inter_list.append(x_inter)
            x_inter = torch.stack(x_inter_list, dim=-2) # (batch, fp_num, in_dim)
        elif self.reduce_inter =='mean':
            x_inter_list = []
            for i in x_inner_list:
                x_inter = x_inner.mean(-2) 
                x_inter_list.append(x_inter)
            x_inter = torch.stack(x_inter_list, dim=-2)   
        elif self.reduce_inter =='max':
            x_inter_list = []
            for i in x_inner_list:
                x_inter = x_inner.max(-2) 
                x_inter_list.append(x_inter)
            x_inter = torch.stack(x_inter_list, dim=-2)
        elif self.reduce_inter =='attn':
            x_inter_list = []
            contributes_inter_list = []
            score_inter_list = []
            for i,gate in enumerate(self.gate_inter):
                x_inner = x_inner_list[i] # (batch, fp_i_num, in_dim)
                mask_per_fp = fp_dense.permute(0,2,1).sum(-1, keepdim= True)[:, fp_range[i][0]:fp_range[i][1], :] < 1e-5 # (batch, fp_i_num, 1)
                score = gate(x_inner) # (batch, fp_i_num, 1) 
                score_mask_bias = mask_per_fp * (-1e12)
                score = score + score_mask_bias
                score_softmax = torch.softmax(score, dim = -2) # (batch, fp_i_num, 1)
                score_inter_list.append(score_softmax.tolist())
                x_inter = score_softmax * x_inner # (batch, fp_i_num, 1) * (batch, fp_i_num, in_dim)
                x_inter = x_inter.sum(-2) # (batch, in_dim)
                x_inter_list.append(x_inter)
                contributes_inter_list.append(score_softmax.unsqueeze(-1)*contributes_inner_list[i]) # [(batch, fp_1, Nmax, 1), (batch, fp_2, Nmax, 1), ...]
            x_inter = torch.stack(x_inter_list, dim=-2) # (batch, fp_num, in_dim) 

        # GLOBAL LEVEL POOLING
        if self.reduce_global =='sum': # (batch ,fp_length, in_dim) -> (batch, in_dim)
            x_global = x_inter.sum(-2) # (batch, in_dim)
        elif self.reduce_global =='mean':
            x_global = x_inter.mean(-2)
        elif self.reduce_global == 'max':
            x_global = x_inter.max(-2)
        elif self.reduce_global == 'attn':
            contributes_global_list = [] 
            contributes_global_list_ = []
            #masks = torch.stack(list(map(lambda a: batch==a, range(size))), dim=0).to(torch.int)
            score = self.gate_global(x_inter) # (batch, fp_num, 1)
            score_softmax = torch.softmax(score, dim=-2)
            x_global = score_softmax * x_inter # (batch, fp_num, 1) * (batch, fp_num, in_dim) -> (batch, fp_num, in_dim)
            for i, contri in enumerate(contributes_inter_list):

                contributes_global_list.append(contri * score_softmax[:, i, :, None, None]) # (batch, fp_i, Nmax, 1) * (batch, 1, 1, 1) -> (batch, fp_i, Nmax, 1)
                contributes_global_list_.append((contri * score_softmax[:, i, :, None, None]).tolist())
            x_global = x_global.sum(-2)
        contributes_global = torch.cat(contributes_global_list, dim=1).sum(dim=1)

        
        return x_global, (contributes_inner_list, contributes_inter_list, contributes_global_list_), (x_dense_mask.tolist(), 
                fp_dense.permute(0,2,1).tolist(), fp_dense.permute(0,2,1).sum(dim=-1).tolist(), fp_length.tolist(), mask_per_fp.tolist(),score_inter_list,score_softmax.tolist(),(score_softmax * x_inter).tolist())


                    