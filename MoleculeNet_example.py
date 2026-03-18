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


from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import ModuleList, Sequential, Linear, Module
import torch.nn.functional as F
from torch_geometric.nn.aggr import Aggregation
from torch_geometric.nn.inits import reset
from torch_geometric.utils import softmax, to_dense_batch

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

Dataset = MoleculeNet2
molecule_names = MoleculeNet2.names.keys()


dataset_name_list = ['bace','bbbp','hiv']
task_num_dict = {'bace':2,'bbbp':2,'hiv':2,'clintox':2,'muv':17,'sider':27,'tox21':12,'toxcast':617,'esol':1,'freesolv':1,'lipo':1}
FPs = ['MorganFP', 'EstateFP', 'FragmentFP', 'MACCSFP', 'PubChemFP', 'RdkitFP', 'RGroupFP']
FP_args_list = [{'nBits':1024, 'radius':2}, {}, {}, {}, {}, {'nBits':1024, 'minPath':1, 'maxPath':5}, {'nBits':1024}]
FP_args_dict = {FPs[i]:FP_args_list[i] for i in range(len(FPs))}
FP_length_dict = {'EstateFP':79, 'FragmentFP':86, 'MACCSFP':166, 'PubChemFP':881, 'RdkitFP':1024, 'RGroupFP':2048, 'MorganFP':1024, }
# fixed model args
fixed_pub_args = {'in_channels': 39,  #in_channels = pre_transform.in_channels
            'out_channels': 1,
            'edge_dim': 10, #in_channels = pre_transform.edge_dim
            'convs_layers': [128, 128],
            'dense_layers': [512, 128, 32],
            'dropout_p': 0.0, 'batch_norms': None}  # ,torch.nn.BatchNorm1d
fixed_train_args = { 'seed': 42,
                     'weight_decay': 1e-4,
                     'epochs': 200, 
                     'device': torch.device('cuda:5')}

baselines = ['sum', 'max', 'mean', 'set2set', 'topkpool', 'sagpool', 'diffpool', 'mincutpool']
#baselines = ['mincutpool']

def trainer():
    device = fixed_train_args['device']
    lr = 1e-4
    bs = 128
    epochs = 200

    for zz, dataset_name in enumerate(dataset_name_list):
        history = {}

        for fp in FPs:
            pre_transform = featurizer.GenNodeEdgeFeatures39_WithFP(fp_types=fp.split('+'))
            dataset = MoleculeNet2(root = '/mnt/cc/0_0datasets/fppooling', name = f'{dataset_name}-{fp}', pre_transform = pre_transform)
            history[fp] = {}
            fixed_pub_args.update({'pooling_layer': pooling.FingerprintPool(in_channel=128, hidden_channel=512, fp_length=torch.tensor([FP_length_dict[fp_temp] for fp_temp in fp.split('+')]), reduce_inner='attn', reduce_inter='attn', reduce_global='attn')})
            fixed_pub_args.update({'fp_dim':[FP_length_dict[fp_temp] for fp_temp in fp.split('+')]})

            # SCAFFOLD SPLIT
            from fppcode import splitters
            for fold, seed in enumerate([0, 42, 100]):
                train_dataset, valid_dataset, test_dataset = splitters.random_scaffold_split(dataset, dataset.smiles, task_idx=None, null_value=0, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=seed)
                history[fp][fold] = []

                random.seed(seed)
                os.environ['PYTHONHASHSEED'] = str(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True

                train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, drop_last=True)
                test_loader = DataLoader(test_dataset, batch_size=32, drop_last=True) 
                
                fixed_pub_args['out_channels'] = task_num_dict[dataset_name]
                gnn_model = model.GNN_GIN(**fixed_pub_args).to(device)

                optimizer = torch.optim.Adam(gnn_model.parameters(), lr=lr, weight_decay= fixed_train_args['weight_decay'])
                
                aca_loss = loss.ACALoss(alpha=0, cliff_lower=1, cliff_upper=1,
                            dev_mode=False, squared=True)

                if dataset_name in ['bace', 'bbbp', 'hiv']:
                    best_roc = 0  
                    for epoch in range(1, epochs):
                        train_cr = train_cls_celoss(train_loader, gnn_model, optimizer, None, fixed_train_args)
                        test_roc = test_2cls_celoss(test_loader, gnn_model, fixed_train_args)
                        if test_roc > best_roc:
                            best_roc = test_roc
                        print(f'Dataset: {dataset_name} ,Epoch: {epoch:03d}, Loss: {train_cr:.4f} Test: {test_roc:.4f} Best: {best_roc:.4f} FP: {fp} Fold:{fold}')
                        history[fp][fold].append(
                            {str(fold) + '_' + 'Epoch': epoch, str(fold) + '_' + 'train_cr': train_cr, str(fold) + '_' + 'test_roc': test_roc, str(fold) + '_' + 'best_roc': best_roc, 'best_roc': best_roc})
                        #wandb.log({dataset_name + '_' + fp + '_' + str(fold) + '_' + 'Epoch':epoch, dataset_name + '_' + fp + '_' + str(fold) + '_' + 'Loss':train_cr, dataset_name + '_' + fp + '_' + str(fold) + '_' + 'Test_roc': test_roc, dataset_name + '_' + fp + '_' + str(fold) + '_' + 'Best_roc': best_roc})
                elif dataset_name in ['clintox','muv','sider','tox21','toxcast']:
                    best_global_roc_mi = 0
                    best_global_roc_ma = 0
                    best_roc_mi = 0  
                    best_roc_ma = 0
                    for epoch in range(1, epochs):
                        train_roc = train_muti2cls_bceloss(train_loader, gnn_model, optimizer, None, fixed_train_args)
                        test_roc_mi, test_roc_ma, test_global_roc_mi, test_global_roc_ma, test_celoss = test_muti2cls_bceloss(test_loader, gnn_model, fixed_train_args)
                        if test_roc_mi > best_roc_mi:
                            best_roc_mi = test_roc_mi
                        if test_roc_ma > best_roc_ma:
                            best_roc_ma = test_roc_ma
                        if test_global_roc_mi > best_global_roc_mi:
                            best_global_roc_mi = test_global_roc_mi
                        if test_global_roc_ma > best_global_roc_ma:
                            best_global_roc_ma = test_global_roc_ma
                        print(f'Epoch: {epoch:03d}, Loss: {train_roc:.4f} Test_mi: {test_roc_mi:.4f} Test_ma: {test_roc_ma:.4f} Test_Global_mi: {test_global_roc_mi:.4f} Test_Global_ma: {test_global_roc_ma:.4f}' \
                            f' Test_CE: {test_celoss:.4f} Best_mi: {best_roc_mi:.4f} Best_ma: {best_roc_ma:.4f} Best_Global_mi: {best_global_roc_mi:.4f} Best_Global_ma: {best_global_roc_ma:.4f} FP: {fp} Fold:{fold}')
                        history[fp][fold].append(
                            {str(fold) + '_' + 'Epoch': epoch, str(fold) + '_' + 'train_roc': train_roc, str(fold) + '_' + 'test_roc_mi': test_roc_mi, str(fold) + '_' + 'test_roc_ma': test_roc_ma,\
                            str(fold) + '_' + 'best_roc_mi': best_roc_mi, 'best_roc_mi': best_roc_mi, 'best_roc_ma': best_roc_ma, 'best_global_roc_mi': best_global_roc_mi,\
                                'best_global_roc_ma': best_global_roc_ma})
                        # wandb.log({fp + '_' + str(fold) + '_' + 'Epoch':epoch, fp + '_' + str(fold) + '_' + 'Loss':train_roc, fp + '_' + str(fold) + '_' + 'Test_roc_mi': test_roc_mi, '_' + str(fold) + '_' + 'Test_roc_ma': test_roc_ma, \
                        # fp + '_' + str(fold) + '_' + 'Best_roc_mi': best_roc_mi, 'Best_roc_ma': best_roc_ma, 'Best_roc_ma': best_roc_ma, 'Best_global_roc_mi': best_global_roc_mi,\
                        # 'Best_global_roc_ma': best_global_roc_ma})
                else:
                    best_rmse = 1e5  
                    for epoch in range(1, epochs):
                        train_rmse = train_reg_mse(train_loader, gnn_model, optimizer, aca_loss, fixed_train_args)
                        test_rmse = test_reg_mse(test_loader, gnn_model, fixed_train_args)
                        if test_rmse < best_rmse:
                            best_rmse = test_rmse
                        print(f'Dataset: {dataset_name} ,Epoch: {epoch:03d}, Loss: {train_rmse:.4f} Test: {test_rmse:.4f} Best: {best_rmse:.4f} Baseline: {fp} Fold:{fold}')
                        history[fp][fold].append(
                            {str(fold) + '_' + 'Epoch': epoch, str(fold) + '_' + 'train_rmse': train_rmse, str(fold) + '_' + 'test_rmse': test_rmse, str(fold) + '_' + 'best_rmse': best_rmse, 'best_rmse': best_rmse})
                        #wandb.log({dataset_name + '_' + fp + '_' + str(fold) + '_' + 'Epoch':epoch, fp + '_' + str(fold) + '_' + 'Loss':train_rmse, dataset_name + '_' + fp + '_' + str(fold) + '_' + 'Test_rmse': test_rmse, dataset_name + '_' + fp  + '_' + 'Best_rmse'+ '_' + str(fold): best_rmse})
            if dataset_name in ['bace', 'bbbp', 'hiv']:
                sum_result = 0        
                for i in range(3):
                    sum_result = sum_result + history[fp][i][-1]['best_roc']
                mean_result = sum_result/3
                print(f'{dataset_name}_{fp}_MEAN_BEST:{mean_result}')
                #wandb.log({dataset_name + '_' + fp + '_' + 'MEAN_BEST':mean_result})
            elif dataset_name in ['clintox','muv','sider','tox21','toxcast']:
                sum_result_mi = 0
                sum_result_ma = 0 
                sum_result_global_mi = 0
                sum_result_global_ma = 0  
                for i in range(3):
                    sum_result_mi = sum_result_mi + history[fp][i][-1]['best_roc_mi']
                    sum_result_ma = sum_result_ma + history[fp][i][-1]['best_roc_ma']
                    sum_result_global_mi = sum_result_global_mi + history[fp][i][-1]['best_global_roc_mi']
                    sum_result_global_ma = sum_result_global_ma + history[fp][i][-1]['best_global_roc_ma']
                mean_result_mi = sum_result_mi/3
                mean_result_ma = sum_result_ma/3
                mean_result_global_mi = sum_result_global_mi/3
                mean_result_global_ma = sum_result_global_ma/3
                print(f'{dataset_name}_{fp}_MEAN_BEST_mi:{mean_result_mi}')
                print(f'{dataset_name}_{fp}_MEAN_BEST_ma:{mean_result_ma}')
                print(f'{dataset_name}_{fp}_MEAN_BEST_global_mi:{mean_result_global_mi}')
                print(f'{dataset_name}_{fp}_MEAN_BEST_global_ma:{mean_result_global_ma}')
                # wandb.log({fp + '_' + 'MEAN_BEST_mi':mean_result_mi, fp + '_' + 'MEAN_BEST_mi':mean_result_ma, \
                #         fp + '_' + 'MEAN_BEST_Global_mi':mean_result_global_mi, fp + '_' + 'MEAN_BEST_Global_ma':mean_result_global_ma})
                        
            else:
                sum_result = 0        
                for i in range(3):
                    sum_result = sum_result + history[fp][i][-1]['best_rmse']
                mean_result = sum_result/3
                print(f'{dataset_name}_{fp}_MEAN_BEST:{mean_result}')
                # wandb.log({dataset_name + '_' + fp + '_' + 'MEAN_BEST':mean_result})




if __name__=='__main__':
    trainer()  
  