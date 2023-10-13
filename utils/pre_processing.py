#!usr/bin/env python3
# -*- coding:utf-8 -*-
# @time : 2022/4/1 12:23
# @author : Xujun Zhang
import argparse
import os
import sys
from multiprocessing import Pool
import numpy as np
from prody import parsePDB, writePDB
from rdkit import Chem
pwd_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(pwd_dir)
sys.path.append(os.path.dirname(pwd_dir))




def prepare_pro(pdb_id):
    '''
    prepare protein structure using schrodinger
    :param pdb_id:
    :return:
    '''
    path_local = f'{dataset_path}/{pdb_id}'
    # src
    raw_protein_file = f'{path_local}/{pdb_id}_protein.pdb'
    # dst
    dst_protein_file = f'{path_local}/{pdb_id}_protein_pred.pdb'
    if not os.path.exists(dst_protein_file) or os.path.getsize(dst_protein_file) == 0:
        cmd = f'cd {path_local} &&' \
              f'module load schrodinger/2022-1 &&' \
              f'timeout 30m prepwizard -rehtreat -watdist 0 -disulfides -mse -fillsidechains -propka_pH 7 ' \
              f'-minimize_adj_h -epik_pH 7 -epik_pHt 0 -preserve_st_titles -NOJOBID {raw_protein_file} {dst_protein_file}'
        os.system(cmd)


def get_pocket_pure(protein_file, somepoint, out_file, size=12):
    protein_prody_obj = parsePDB(protein_file)
    condition = f'same residue as exwithin {size} of somepoint'
    pocket_selected = protein_prody_obj.select(condition,
                                               somepoint=somepoint)  # （n, 3）
    writePDB(out_file, atoms=pocket_selected)


def get_mol2_xyz_from_cmd(ligand_mol2):
    x = os.popen(
        "cat %s | sed -n '/@<TRIPOS>ATOM/,/@<TRIPOS>BOND/'p | awk '{print $3}'" % ligand_mol2).read().splitlines()[1:-1]
    y = os.popen(
        "cat %s | sed -n '/@<TRIPOS>ATOM/,/@<TRIPOS>BOND/'p | awk '{print $4}'" % ligand_mol2).read().splitlines()[1:-1]
    z = os.popen(
        "cat %s | sed -n '/@<TRIPOS>ATOM/,/@<TRIPOS>BOND/'p | awk '{print $5}'" % ligand_mol2).read().splitlines()[1:-1]
    return np.asanyarray(list(zip(x, y, z))).astype(float)


def get_pocket(pdb_id):
    '''
    get binding pocket based on reference crystal ligand
    and move random generate liand pose to the center of pocket
    :param pdb_id:
    :return:
    '''
    path_local = f'{dataset_path}/{pdb_id}'
    # src
    protein_pdb = f'{path_local}/{pdb_id}_protein.pdb'
    reference_ligand_sdf = f'{path_local}/{pdb_id}_ligand.sdf'
    reference_ligand_mol2 = f'{path_local}/{pdb_id}_ligand.mol2'
    # dst
    pocket_pdb = f'{path_local}/{pdb_id}_pocket_ligH12A.pdb'
    # select & save pocket
    ligand_mol = Chem.MolFromMolFile(reference_ligand_sdf, removeHs=False)
    if ligand_mol is None:
        ligand_mol = Chem.MolFromMol2File(
            reference_ligand_mol2, removeHs=False)
    if ligand_mol is None:
        try:
            ligpos = get_mol2_xyz_from_cmd(reference_ligand_mol2)
        except:
            print(pdb_id)
    else:
        ligpos = ligand_mol.GetConformer().GetPositions()
    if not os.path.exists(pocket_pdb) or os.path.getsize(pocket_pdb) == 0:
        protein_prody_obj = parsePDB(protein_pdb)
        condition = 'same residue as exwithin 12 of somepoint'
        pocket_selected = protein_prody_obj.select(condition,
                                                   somepoint=ligpos)  # （n, 3）
        writePDB(pocket_pdb, atoms=pocket_selected)


def pipeline(pdb_id):
    try:
        get_pocket(pdb_id)
    except:
        print(f'{pdb_id} error')


if __name__ == '__main__':
    # get args
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--complex_file_dir', type=str, default='')
    args = argparser.parse_args()
    # 
    dataset_path = args.complex_file_dir
    pdb_ids = os.listdir(dataset_path)
    # multiprocessing
    pool = Pool()
    pool.map(pipeline, pdb_ids)
    pool.close()
    pool.join()
