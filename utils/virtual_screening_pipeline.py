#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   virtual_screening_pipeline.py
@Time    :   2023/10/13 11:25:33
@Author  :   Xujun Zhang
@Version :   1.0
@Contact :   
@License :   
@Desc    :   None
'''
# here put the import lib
import os
import pandas as pd
from tqdm import tqdm
import argparse
import os
import sys

import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import pandas as pd
import torch.optim
# from prefetch_generator import BackgroundGenerator
from tqdm import tqdm

# dir of current
project_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(project_dir)
from architecture.KarmaDock_architecture import KarmaDock
from utils.post_processing import correct_pos
from dataset.graph_obj import *
from utils.pre_processing import get_pocket_pure
from utils.fns import Early_stopper, set_random_seed
from dataset.dataloader_obj import PassNoneDataLoader
argparser = argparse.ArgumentParser()
argparser.add_argument('--mode', default='vs', help='virtual screening on GPU or generating graph on CPUs')
argparser.add_argument('--ligand_smi', type=str,
                       default='/root/chemdiv.smi',
                       help='the smiles file for virtual screening')
argparser.add_argument('--protein_file', type=str,
                       default='/root/KarmaDock/DEKOIS2/pdk1/protein/pdk1_protein.pdb',
                       help='the protein files path')
argparser.add_argument('--crystal_ligand_file', type=str,
                       default='/root/KarmaDock/DEKOIS2/pdk1/protein/pdk1_ligand.mol2 ',
                       help='the crystal ligand files path for binding site locating')
argparser.add_argument('--graph_dir', type=str,
                       default='/root/graphs',
                       help='the dir for saving graph')
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


# set random seed
set_random_seed(args.random_seed)
# get data 
vs_libray_path = args.graph_dir
protein_file = args.protein_file
pocket_file = args.protein_file.replace('.pdb', '_pocket.pdb')
ligand_file = args.crystal_ligand_file
score_threshold = args.score_threshold
# get pocket center
cry_lig_pos = get_mol2_xyz_from_cmd(args.crystal_ligand_file)
pocket_center = torch.from_numpy(cry_lig_pos).to(torch.float32).mean(dim=0)
# get pocket 
if not os.path.exists(pocket_file):
    get_pocket_pure(protein_file, cry_lig_pos, pocket_file, size=12)
# test 
test_dataset = VSTestGraphDataset_FlyReload_SMI(protein_file=pocket_file, ligand_path=vs_libray_path, pocket_center=pocket_center)
if args.mode != 'vs':
    print('# generate graph')
    test_dataset.generate_graphs(ligand_smi=args.ligand_smi, n_job=-1)
else:
    print('# virtual screening')
    os.makedirs(args.out_dir, exist_ok=True)
    # my_device = 'cpu'
    if torch.cuda.is_available():
        device_id = 0
        my_device = f'cuda:{device_id}'
    # model
    model = KarmaDock()
    model = nn.DataParallel(model, device_ids=[device_id], output_device=device_id)
    model.to(my_device)
    # stoper
    model_file = f'{project_dir}/trained_models/karmadock_screening.pkl'
    stopper = Early_stopper(model_file=model_file,
                            mode='lower', patience=70)
    # load existing model
    stopper.load_model(model_obj=model, my_device=my_device, strict=False)
    print('# load model')
    dst_csv = f'{args.out_dir}/score.csv'
    # dataloader
    test_dataloader = PassNoneDataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, 
                                num_workers=0, 
                                follow_batch=[], 
                                pin_memory=True, 
                                # prefetch_factor=2, 
                                # persistent_workers=True
                                )
    print(f'dataset num: {len(test_dataset)}')
    if len(test_dataset) > 0:
        model.eval()
        pki_scores_pred = torch.as_tensor([]).to(my_device)
        pki_scores_pred_ff = torch.as_tensor([]).to(my_device)
        pki_scores_pred_aligned = torch.as_tensor([]).to(my_device)
        pdb_ids = []
        with torch.no_grad():
            for idx, data in enumerate(tqdm(test_dataloader, desc='prediction')):
                # try:
                # to device
                data = data.to(my_device)
                batch_size = data['ligand'].batch[-1] + 1
                # forward
                pro_node_s, lig_node_s = model.module.encoding(data)
                lig_pos, _, _ = model.module.docking(pro_node_s, lig_node_s, data, recycle_num=3)
                mdn_score_pred = model.module.scoring(lig_s=lig_node_s, lig_pos=lig_pos, pro_s=pro_node_s, data=data,
                                                                            dist_threhold=5., batch_size=batch_size)
                pki_scores_pred = torch.cat([pki_scores_pred, mdn_score_pred], dim=0)
                pdb_ids.extend(data.pdb_id)
                # # post processing
                data.pos_preds = lig_pos
                poses, _, _ = correct_pos(data, mask=mdn_score_pred <= score_threshold, out_dir=args.out_dir, out_init=args.out_init, out_uncoorected=args.out_uncoorected, out_corrected=args.out_corrected)
                ff_corrected_pos = torch.from_numpy(np.concatenate([i[0] for i in poses], axis=0)).to(my_device)
                align_corrected_pos = torch.from_numpy(np.concatenate([i[1] for i in poses], axis=0)).to(my_device)
                mdn_score_pred_ff_corrected = model.module.scoring(lig_s=lig_node_s, lig_pos=ff_corrected_pos, pro_s=pro_node_s, data=data,
                                                                        dist_threhold=5., batch_size=batch_size)
                mdn_score_pred_align_corrected = model.module.scoring(lig_s=lig_node_s, lig_pos=align_corrected_pos, pro_s=pro_node_s, data=data,
                                                                    dist_threhold=5., batch_size=batch_size)
                pki_scores_pred_ff = torch.cat([pki_scores_pred_ff, mdn_score_pred_ff_corrected], dim=0)
                pki_scores_pred_aligned = torch.cat([pki_scores_pred_aligned, mdn_score_pred_align_corrected], dim=0)
                # except:
                #     continue
        pki_scores_pred = pki_scores_pred.view(-1).cpu().numpy().tolist()
        pki_scores_pred_ff = pki_scores_pred_ff.view(-1).cpu().numpy().tolist()
        pki_scores_pred_aligned = pki_scores_pred_aligned.view(-1).cpu().numpy().tolist()
        data = zip(pdb_ids, pki_scores_pred, pki_scores_pred_ff, pki_scores_pred_aligned) # pki_scores_pred_ff, pki_scores_pred_aligned
        columnds = ['pdb_id', 'karma_score', 'karma_score_ff', 'karma_score_aligned']
        df = pd.DataFrame(data, columns=columnds)
        df.to_csv(dst_csv, index=False)