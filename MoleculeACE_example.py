import sys
sys.path.insert(0,'..')
import torch_geometric
from torch_geometric.nn.models import AttentiveFP
from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader
from rdkit import Chem
import torch.nn.functional as F
import torch
import pandas as pd
from math import sqrt
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import os
import random
#import pytorch_lightning

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import ModuleList, Sequential, Linear, Module
import torch.nn.functional as F
from torch_geometric.nn.aggr import Aggregation
from torch_geometric.nn.inits import reset
from torch_geometric.utils import softmax

from sklearn.metrics import roc_auc_score

from fppcode import loss,model,pooling,saver
from fppcode.feature import featurizer
from fppcode.dataset import LSSNS, HSSMS, MoleculeNet2  # dataset
from fppcode.feature.featurizer import GenNodeEdgeFeatures115, GenNodeEdgeFeatures39, GenNodeEdgeFeatures39_WithFP, transform_fp, ChangeFP_Dataset  # feature
from fppcode.model import GNN_GCN, GNN_GIN, GNN_GAT, GNN_PNA, get_deg  # model
from fppcode.pooling import SubstructurePool
from fppcode.loss import ACALoss
from fppcode.saver import SaveBestModel

from fppcode.trains import train_reg_mse, train_cls_celoss, train_muti2cls_bceloss, test_2cls_celoss, test_reg_mse, test_muti2cls_bceloss
from fppcode.constants import FPs, FP_length_dict, FP_args_list, FP_args_dict

# Dataset = MoleculeNet2
# molecule_names = MoleculeNet2.names.keys()

all_dataset_names = ['chembl1871_ki',
 'chembl218_ec50',
 'chembl244_ki',
 'chembl236_ki',
 'chembl234_ki',
 'chembl219_ki',
 'chembl238_ki',
 'chembl4203_ki',
 'chembl2047_ec50',
 'chembl4616_ec50',
 'chembl2034_ki',
 'chembl262_ki',
 'chembl231_ki',
 'chembl264_ki',
 'chembl2835_ki',
 'chembl2971_ki',
 'chembl237_ec50',
 'chembl237_ki',
 'chembl4792_ki',
 'chembl239_ec50',
 'chembl3979_ec50',
 'chembl235_ec50',
 'chembl4005_ki',
 'chembl2147_ki',
 'chembl214_ki',
 'chembl228_ki',
 'chembl287_ki',
 'chembl204_ki',
 'chembl1862_ki',
 'chembl233_ki']

dataset_name_list = ['chembl1871_ki', 'chembl218_ec50', 'chembl244_ki', 'chembl236_ki', 'chembl234_ki', 'chembl219_ki']

FPs = ['MorganFP', 'EstateFP', 'FragmentFP', 'MACCSFP', 'PubChemFP', 'RdkitFP', 'RGroupFP']
FP_args_list = [{'nBits':1024, 'radius':2}, {}, {}, {}, {}, {'nBits':1024, 'minPath':1, 'maxPath':5}, {'nBits':1024}]
FP_args_dict = {FPs[i]:FP_args_list[i] for i in range(len(FPs))}
FP_length_dict = {'MorganFP':1024, 'EstateFP':79, 'FragmentFP':86, 'MACCSFP':166, 'PubChemFP':881, 'RdkitFP':1024, 'RGroupFP':2048}
# fixed model args
fixed_pub_args = {'in_channels': 39,  #in_channels = pre_transform.in_channels
            'out_channels': 1,
            'edge_dim': 10, #in_channels = pre_transform.edge_dim
            'convs_layers': [128, 32],
            'dense_layers': [512, 128, 32],
            'dropout_p': 0.0, 'batch_norms': None}  # ,torch.nn.BatchNorm1d
fixed_train_args = { 'seed': 42,
                     'weight_decay': 1e-4,
                     'epochs': 200, 
                     'device': torch.device('cuda:5')}


# Set random seed
seed = fixed_train_args['seed']
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def trainer():

    device = fixed_train_args['device']

    lr = 1e-4
    bs = 8
    epochs = 200
    for dataset_name in dataset_name_list:
        history = {}
        for fp in FPs:
            history[fp] = {}
            fixed_pub_args.update({'pooling_layer': pooling.FingerprintPool(in_channel=32, hidden_channel=512, fp_length=torch.tensor([FP_length_dict[fp]]), reduce_inner='attn', reduce_inter='attn', reduce_global='attn')})
            fixed_pub_args.update({'fp_dim':[FP_length_dict[fp]]})
            
            pre_transform = featurizer.GenNodeEdgeFeatures39_WithFP(fp_types=[fp])
            dataset = HSSMS (root = '/mnt/cc/0_0datasets/fppooling', name = f'{dataset_name}-{fp}', pre_transform = pre_transform)
            

            from fppcode import splitters
            for fold, seed in enumerate([0, 42, 100]):
                train_dataset, valid_dataset, test_dataset = splitters.random_scaffold_split(dataset, dataset.smiles, task_idx=None, null_value=0, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=seed)
                history[fp][fold] = []

                train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
                test_loader = DataLoader(test_dataset, batch_size=32) 
                

                gnn_model = model.GNN_GIN(**fixed_pub_args).to(device)

                #optimizer = torch.optim.Adam(gnn_model.parameters(), lr=train_args.lr, weight_decay= train_args.weight_decay)
                optimizer = torch.optim.Adam(gnn_model.parameters(), lr=lr, weight_decay= fixed_train_args['weight_decay'])
                
                aca_loss = loss.ACALoss(alpha=0, cliff_lower=1, cliff_upper=1,
                            dev_mode=False, squared=True)

                best_rmse = 1e5  
                #import ipdb;ipdb.set_trace()
                for epoch in range(1, epochs):
                    train_rmse = train_reg_mse(train_loader, gnn_model, optimizer, aca_loss, fixed_train_args)
                    test_rmse = test_reg_mse(test_loader, gnn_model, fixed_train_args)
                    if test_rmse < best_rmse:
                        best_rmse = test_rmse
                    print(f'Dataset: {dataset_name} ,Epoch: {epoch:03d}, Loss: {train_rmse:.4f} Test: {test_rmse:.4f} Best: {best_rmse:.4f} FP: {fp} Fold:{fold}')
                    history[fp][fold].append(
                        {str(fold) + '_' + 'Epoch': epoch, str(fold) + '_' + 'train_rmse': train_rmse, str(fold) + '_' + 'test_rmse': test_rmse, str(fold) + '_' + 'best_rmse': best_rmse, 'best_rmse': best_rmse})
                    #wandb.log({dataset_name + '_' + fp + '_' + str(fold) + '_' + 'Epoch':epoch, fp + '_' + str(fold) + '_' + 'Loss':train_rmse, dataset_name + '_' + fp + '_' + str(fold) + '_' + 'Test_rmse': test_rmse, dataset_name + '_' + fp  + '_' + 'Best_rmse'+ '_' + str(fold): best_rmse})
            sum_result = 0        
            for i in range(3):
                sum_result = sum_result + history[fp][i][-1]['best_rmse']
            mean_result = sum_result/3
            print(f'{dataset_name}_{fp}_MEAN_BEST:{mean_result}')
            #wandb.log({dataset_name + '_' + fp + '_' + 'MEAN_BEST':mean_result})



if __name__=='__main__':
    trainer()  
  