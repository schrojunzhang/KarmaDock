#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   virtual_screening.py
@Time    :   2023/07/04 12:14:21
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

import numpy as np
import torch.nn as nn
import pandas as pd
import torch.optim
from prefetch_generator import BackgroundGenerator
import warnings
from tqdm import tqdm
# dir of current
warnings.filterwarnings('ignore')
project_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(project_dir)
from utils.fns import Early_stopper, set_random_seed
from dataset.graph_obj import VSTestGraphDataset_Fly_SMI, get_mol2_xyz_from_cmd
from dataset.dataloader_obj import PassNoneDataLoader
from architecture.KarmaDock_architecture import KarmaDock
from utils.post_processing import correct_pos
from pre_processing import get_pocket_pure

class DataLoaderX(PassNoneDataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

# get parameters from command line
argparser = argparse.ArgumentParser()
argparser.add_argument('--ligand_smi', type=str,
                       default='/root/KarmaDock/DEKOIS2/pdk1/active_decoys.smi',
                       help='the ligand smiles path')
argparser.add_argument('--protein_file', type=str,
                       default='/root/KarmaDock/DEKOIS2/pdk1/protein/pdk1_protein.pdb',
                       help='the protein files path')
argparser.add_argument('--crystal_ligand_file', type=str,
                       default='/root/KarmaDock/DEKOIS2/pdk1/protein/pdk1_ligand.mol2 ',
                       help='the crystal ligand files path for binding site locating')
argparser.add_argument('--out_dir', type=str,
                       default='/root/KarmaDock/DEKOIS2/pdk1/karmadocked',
                       help='dir for recording binding poses and binding scores')
argparser.add_argument('--score_threshold', type=float, 
                       default=70,
                       help='score threshold for saving binding poses')
argparser.add_argument('--batch_size', type=int,
                       default=64,
                       help='batch size')
argparser.add_argument('--random_seed', type=int,
                       default=2020,
                       help='random_seed')
argparser.add_argument('--out_init', action='store_true', default=False, help='whether to save initial poses to sdf file')
argparser.add_argument('--out_uncoorected', action='store_true', default=False, help='whether to save uncorrected poses to sdf file')
argparser.add_argument('--out_corrected', action='store_true', default=False, help='whether to save corrected poses to sdf file')

args = argparser.parse_args()
set_random_seed(args.random_seed)
os.makedirs(args.out_dir, exist_ok=True)
pocket_file = args.protein_file.replace('.pdb', '_pocket.pdb')
cry_lig_pos = get_mol2_xyz_from_cmd(args.crystal_ligand_file)
pocket_center = cry_lig_pos.mean(axis=0)
if not os.path.exists(pocket_file):
    get_pocket_pure(args.protein_file, somepoint=cry_lig_pos, out_file=pocket_file, size=12)
# dataset
test_dataset = VSTestGraphDataset_Fly_SMI(protein_file=pocket_file, ligand_path=args.ligand_smi, pocket_center=pocket_center)
# dataloader
test_dataloader = DataLoaderX(dataset=test_dataset, batch_size=args.batch_size, shuffle=False,
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
model_file = f'{project_dir}/trained_models/karmadock_screening.pkl'
stopper = Early_stopper(model_file=model_file,
                        mode='lower', patience=10)
print('# load model')
# load existing model
stopper.load_model(model_obj=model, my_device=my_device, strict=False)
# time
pdb_ids = []
# labels = []
binding_scores = []
binding_scores_ff_corrected = []
binding_scores_align_corrected = []
with torch.no_grad():
    model.eval()
    for idx, data in enumerate(tqdm(test_dataloader)):
        # to device
        data = data.to(my_device)
        batch_size = data['ligand'].batch[-1] + 1
        # forward
        pro_node_s, lig_node_s = model.module.encoding(data)
        lig_pos, _, _ = model.module.docking(pro_node_s, lig_node_s, data, recycle_num=3)
        mdn_score_pred = model.module.scoring(lig_s=lig_node_s, lig_pos=lig_pos, pro_s=pro_node_s, data=data,
                                                                    dist_threhold=5., batch_size=batch_size)
        # post  processing
        data.pos_preds = lig_pos
        poses, _, _ = correct_pos(data, mask=mdn_score_pred <= args.score_threshold, out_dir=args.out_dir, out_init=args.out_init, out_uncoorected=args.out_uncoorected, out_corrected=args.out_corrected)
        ff_corrected_pos = torch.from_numpy(np.concatenate([i[0] for i in poses], axis=0)).to(my_device)
        align_corrected_pos = torch.from_numpy(np.concatenate([i[1] for i in poses], axis=0)).to(my_device)
        mdn_score_pred_ff_corrected = model.module.scoring(lig_s=lig_node_s, lig_pos=ff_corrected_pos, pro_s=pro_node_s, data=data,
                                                                dist_threhold=5., batch_size=batch_size)
        mdn_score_pred_align_corrected = model.module.scoring(lig_s=lig_node_s, lig_pos=align_corrected_pos, pro_s=pro_node_s, data=data,
                                                            dist_threhold=5., batch_size=batch_size)
        binding_scores_align_corrected.extend(mdn_score_pred_align_corrected.cpu().numpy().tolist())
        binding_scores_ff_corrected.extend(mdn_score_pred_ff_corrected.cpu().numpy().tolist())
        pdb_ids.extend(data.pdb_id)
        binding_scores.extend(mdn_score_pred.cpu().numpy().tolist())
        # labels.extend([1 if not i.startswith('ZINC') else 0 for i in data.pdb_id])
    # out to csv
    df_score = pd.DataFrame(list(zip(pdb_ids, binding_scores)), columns=['pdb_id', 'karma_score'])
    df_score['karma_score_ff'] = binding_scores_ff_corrected
    df_score['karma_score_aligned'] = binding_scores_align_corrected
    df_score.to_csv(f'{args.out_dir}/score.csv', index=False)