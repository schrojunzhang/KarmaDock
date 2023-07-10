#!usr/bin/env python3
# -*- coding:utf-8 -*-
# @time : 2022/8/13 13:18
# @author : Xujun Zhang

import os
import sys
import argparse
import  numpy as np
import pandas as pd

pwd_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(pwd_dir))
from dataset import graph_obj

# get parameters from command line
argparser = argparse.ArgumentParser()
argparser.add_argument('--complex_file_dir', type=str,
                       default='/root/project_7/data/sc_complexes',
                       help='the complex file path')
argparser.add_argument('--graph_file_dir', type=str,
                       default='/root/project_7/data/graphs/test',
                       help='the graph files path')
args = argparser.parse_args()
os.makedirs(args.graph_file_dir, exist_ok=True)
pdb_ids = os.listdir(args.complex_file_dir)
# generate graph
test_dataset = graph_obj.PDBBindGraphDataset(src_dir=args.complex_file_dir,
                                        dst_dir=args.graph_file_dir,
                                        pdb_ids=pdb_ids,
                                        dataset_type='test',
                                        # n_job=64,
                                        n_job=1,
                                        on_the_fly=True,
                                        verbose=True)
