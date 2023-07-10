#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   ligand_docking.py
@Time    :   2023/03/05 15:16:28
@Author  :   Xujun Zhang
@Version :   1.0
@Contact :   
@License :   
@Desc    :   None
'''

# here put the import lib

import argparse
import os
import sys
import time

import numpy as np
import torch.nn as nn
import pandas as pd
import torch.optim
from torch_geometric.loader import DataLoader
from prefetch_generator import BackgroundGenerator
import rmsd
from tqdm import tqdm
# dir of current
pwd_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(pwd_dir))
from utils.fns import Early_stopper, set_random_seed
from dataset.graph_obj import PDBBindGraphDataset
from dataset.dataloader_obj import PassNoneDataLoader
from architecture.KarmaDock_architecture import KarmaDock
from utils.post_processing import correct_pos

class DataLoaderX(PassNoneDataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

# get parameters from command line
argparser = argparse.ArgumentParser()
argparser.add_argument('--graph_file_dir', type=str,
                       default='/root/KarmaDock/pdbbind_graph',
                       help='the graph files path')
argparser.add_argument('--model_file', type=str,
                       default='/root/KarmaDock/trained_models/karmadock_screening.pkl',
                       help='model file')
argparser.add_argument('--out_dir', type=str,
                       default='/root/KarmaDock/pdbbind_result',
                       help='dir for recording binding poses and binding scores')
argparser.add_argument('--docking', type=str,
                       default='True',
                       help='whether generating binding poses')
argparser.add_argument('--scoring', type=str,
                       default='True',
                       help='whether predict binding affinities')
argparser.add_argument('--correct', type=str,
                       default='True',
                       help='whether correct the predicted binding poses')
argparser.add_argument('--batch_size', type=int,
                       default=64,
                       help='batch size')
argparser.add_argument('--random_seed', type=int,
                       default=2020,
                       help='random_seed')
argparser.add_argument('--csv_file', type=str,
                       default='/root/project_7/data/pdbbind2020.csv',
                       help='the csv file with dataset split')

args = argparser.parse_args()
set_random_seed(args.random_seed)
test_pdb_ids = [ i.split('.')[0] for i in os.listdir(args.graph_file_dir)]
# dataset
test_dataset = PDBBindGraphDataset(src_dir='',
                              dst_dir=args.graph_file_dir,
                              pdb_ids=test_pdb_ids,
                              dataset_type='test',
                              n_job=1,
                              on_the_fly=True)

# dataloader
test_dataloader = DataLoaderX(dataset=test_dataset, batch_size=args.batch_size, shuffle=True,
                             num_workers=0, follow_batch=[], pin_memory=True)
# device
device_id = 0
if torch.cuda.is_available():
    my_device = f'cuda:{device_id}'
else:
    my_device = 'cpu'
# model
model = KarmaDock()
model = nn.DataParallel(model, device_ids=[device_id], output_device=device_id)
model.to(my_device)
# stoper
stopper = Early_stopper(model_file=args.model_file,
                        mode='lower', patience=10)
model_name = args.model_file.split('/')[-1].split('_')[1]
print('# load model')
# load existing model
stopper.load_model(model_obj=model, my_device=my_device, strict=False)
# time
start_time = time.perf_counter()
total_time = 0
data_statistic = []
for re in range(3):
    rmsds = torch.as_tensor([]).to(my_device)
    binding_scores = []
    pdb_ids = []
    ff_corrected_rmsds = []
    align_corrected_rmsds = []
    model.eval()
    egnn_time = 0
    ff_time = 0
    align_time = 0
    out_dir = f'{args.out_dir}/{re}'
    os.makedirs(out_dir, exist_ok=True)
    with torch.no_grad():
        for idx, data in enumerate(tqdm(test_dataloader)):
            # to device
            data = data.to(my_device)
            # forward
            egnn_start_time = time.perf_counter()
            pos_pred, mdn_score = model.module.ligand_docking(data, docking=args.docking, scoring=args.scoring, recycle_num=3, dist_threhold=5)
            egnn_time += time.perf_counter() - egnn_start_time
            pos_true = data['ligand'].xyz
            batch = data['ligand'].batch
            pos_loss = model.module.cal_rmsd(pos_true, pos_pred, batch) 
            rmsds = torch.cat([rmsds, pos_loss], dim=0)
            if args.correct == 'True':
                data.pos_preds = pos_pred
                poses, ff_t, align_t = correct_pos(data, out_dir=out_dir, out_init=False, out_uncoorected=True, out_corrected=True)
                ff_time += ff_t
                align_time += align_t
                ff_corrected_rmsds.extend([rmsd.rmsd(pos_lis[0], pos_lis[2]) for pos_lis in poses])
                align_corrected_rmsds.extend([rmsd.rmsd(pos_lis[1], pos_lis[2]) for pos_lis in poses])
            else:
                real_batch_size = data['ligand'].batch[-1] + 1
                ff_corrected_rmsds.extend([999] * real_batch_size)
                align_corrected_rmsds.extend([999] * real_batch_size)
            pdb_ids.extend(data.pdb_id)
            binding_scores.extend(mdn_score.cpu().numpy().tolist())
        ff_corrected_rmsds = np.asarray(ff_corrected_rmsds)
        align_corrected_rmsds = np.asarray(align_corrected_rmsds)
        # out to csv
        df_score = pd.DataFrame(list(zip(pdb_ids, binding_scores)), columns=['pdb_id', 'score'])
        df_score['RMSD'] = rmsds.cpu().numpy()
        df_score['FF_RMSD'] = ff_corrected_rmsds
        df_score['Aligned_RMSD'] = align_corrected_rmsds
        df_score.to_csv(f'{args.out_dir}/{re}.csv', index=False)
        # statistic
        data_statistic.append([rmsds.mean(), rmsds.median(), (rmsds<=2).sum()/rmsds.size(0), egnn_time / 60, 
                               ff_corrected_rmsds.mean(), np.median(ff_corrected_rmsds), (ff_corrected_rmsds<=2).sum()/ff_corrected_rmsds.shape[0], ff_time / 60, 
                               align_corrected_rmsds.mean(), np.median(align_corrected_rmsds), (align_corrected_rmsds<=2).sum()/align_corrected_rmsds.shape[0], align_time / 60
                               ])
data_statistic_mean = torch.as_tensor(data_statistic).mean(dim=0)
data_statistic_std = torch.as_tensor(data_statistic).std(dim=0)
prediction_time = time.perf_counter()
print(f'''
Total Time: {(prediction_time - start_time) / 60} min
Sample Num: {len(test_dataset)}
# uncorrected
Time Spend: {data_statistic_mean[3]} ± {data_statistic_std[3]} min
Mean RMSD: {data_statistic_mean[0]} ± {data_statistic_std[0]}
Medium RMSD: {data_statistic_mean[1]} ± {data_statistic_std[1]}
Success RATE(2A): {data_statistic_mean[2]} ± {data_statistic_std[2]}
# ff_corrected
Time Spend: {data_statistic_mean[7]} ± {data_statistic_std[7]} min
Mean RMSD: {data_statistic_mean[4]} ± {data_statistic_std[4]}
Medium RMSD: {data_statistic_mean[5]} ± {data_statistic_std[5]}
Success RATE(2A): {data_statistic_mean[6]} ± {data_statistic_std[6]} 
# align corrected
Time Spend: {data_statistic_mean[11]} ± {data_statistic_std[11]} min
Mean RMSD: {data_statistic_mean[8]} ± {data_statistic_std[8]}
Medium RMSD: {data_statistic_mean[9]} ± {data_statistic_std[9]}
Success RATE(2A): {data_statistic_mean[10]} ± {data_statistic_std[10]} 
      ''')
