#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   post_processing.py
@Time    :   2023/02/14 16:14:06
@Author  :   Xujun Zhang
@Version :   1.0
@Contact :   
@License :   
@Desc    :   None
'''

# here put the import lib
import copy
import networkx as nx
import numpy as np
import rmsd
from rdkit import Chem, RDLogger
from rdkit import Geometry
from rdkit.Chem import AllChem, rdMolTransforms
from scipy.optimize import differential_evolution
import time

# part of this code taken from EquiBind https://github.com/HannesStark/EquiBind
RDLogger.DisableLog('rdApp.*')
RMSD = AllChem.AlignMol


def GetDihedral(conf, atom_idx):
    return rdMolTransforms.GetDihedralRad(conf, atom_idx[0], atom_idx[1], atom_idx[2], atom_idx[3])


def SetDihedral(conf, atom_idx, new_vale):
    rdMolTransforms.SetDihedralRad(conf, atom_idx[0], atom_idx[1], atom_idx[2], atom_idx[3], new_vale)

def get_torsion_bonds(mol):
    torsions_list = []
    G = nx.Graph()
    # for i, atom in enumerate(mol.GetAtoms()):
    #     G.add_node(i)
    # nodes = set(G.nodes())
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        G.add_edge(start, end)
    for e in G.edges():
        G2 = copy.deepcopy(G)
        G2.remove_edge(*e)
        if nx.is_connected(G2): continue
        l = list(sorted(nx.connected_components(G2), key=len)[0])
        if len(l) < 2: continue
        n0 = list(G2.neighbors(e[0]))
        n1 = list(G2.neighbors(e[1]))
        torsions_list.append(
            (n0[0], e[0], e[1], n1[0])
        )
    return torsions_list


# GeoMol
def get_torsions(mol_list):
    # print('USING GEOMOL GET TORSIONS FUNCTION')
    atom_counter = 0
    torsionList = []
    for m in mol_list:
        torsionSmarts = '[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]'
        torsionQuery = Chem.MolFromSmarts(torsionSmarts)
        matches = m.GetSubstructMatches(torsionQuery)
        for match in matches:
            idx2 = match[0]
            idx3 = match[1]
            bond = m.GetBondBetweenAtoms(idx2, idx3)
            jAtom = m.GetAtomWithIdx(idx2)
            kAtom = m.GetAtomWithIdx(idx3)
            for b1 in jAtom.GetBonds():
                if (b1.GetIdx() == bond.GetIdx()):
                    continue
                idx1 = b1.GetOtherAtomIdx(idx2)
                for b2 in kAtom.GetBonds():
                    if ((b2.GetIdx() == bond.GetIdx())
                            or (b2.GetIdx() == b1.GetIdx())):
                        continue
                    idx4 = b2.GetOtherAtomIdx(idx3)
                    # skip 3-membered rings
                    if (idx4 == idx1):
                        continue
                    if m.GetAtomWithIdx(idx4).IsInRing():
                        torsionList.append(
                            (idx4 + atom_counter, idx3 + atom_counter, idx2 + atom_counter, idx1 + atom_counter))
                        break
                    else:
                        torsionList.append(
                            (idx1 + atom_counter, idx2 + atom_counter, idx3 + atom_counter, idx4 + atom_counter))
                        break
                break

        atom_counter += m.GetNumAtoms()
    return torsionList


def torsional_align(rdkit_mol, pred_conf, rotable_bonds):
    for rotable_bond in rotable_bonds:
        diheral_angle = GetDihedral(pred_conf, rotable_bond)
        SetDihedral(rdkit_mol.GetConformer(0), rotable_bond, diheral_angle)
    return rdkit_mol

def random_torsion(mol):
    rotable_bonds = get_torsions([mol])
    torsion_updates = np.random.uniform(low=-np.pi, high=np.pi, size=len(rotable_bonds))
    for idx, rotable_bond in enumerate(rotable_bonds):
        SetDihedral(mol.GetConformer(0), rotable_bond, torsion_updates[idx])
    return mol

def mmff_func(mol):
    mol_mmff = copy.deepcopy(mol)
    AllChem.MMFFOptimizeMoleculeConfs(mol_mmff, mmffVariant='MMFF94s')
    for i in range(mol.GetNumConformers()):
        coords = mol_mmff.GetConformers()[i].GetPositions()
        for j in range(coords.shape[0]):
            mol.GetConformer(i).SetAtomPosition(j,
                                                Geometry.Point3D(*coords[j]))

def init_mol_pos(mol):
    feed_back = [1, 1]
    while feed_back[0] == 0:
        feed_back = AllChem.MMFFOptimizeMoleculeConfs(mol, mmffVariant='MMFF94s')
    return mol


def set_rdkit_mol_position(rdkit_mol, position):
    for j in range(position.shape[0]):
        rdkit_mol.GetConformer().SetAtomPosition(j,
                                            Geometry.Point3D(*position[j]))
    return rdkit_mol

def position_align_mol(rdkit_mol, refer_mol):
    rmsd = AllChem.AlignMol(rdkit_mol, refer_mol)
    return rmsd

def position_align_np(rdkit_mol, refer_mol, algo='kabsch'):
    A = rdkit_mol.GetConformer().GetPositions()
    B = refer_mol.GetConformer().GetPositions()
    B_center = rmsd.centroid(B)
    A -= rmsd.centroid(A)
    B -= B_center
    rmsd.quaternion_rotate
    if algo == 'kabsch':
        U = rmsd.kabsch(A, B)
    else: # quaternion
        U = rmsd.quaternion_rotate(A, B)
    A = np.dot(A, U)
    A += B_center
    set_rdkit_mol_position(rdkit_mol=rdkit_mol, position=A)


def correct_pos(data, out_dir, mask=[], out_init=False, out_movie=False, out_uncoorected=True, out_corrected=True, addHs=True):
    poses = []
    # pocket_centers = data.pocket_center.cpu().numpy().astype(np.float64)
    for idx, mol in enumerate(data['ligand'].mol):
        # correct pos
        pos_pred = data.pos_preds[data['ligand'].batch==idx].cpu().numpy().astype(np.float64) # + pocket_centers[idx]
        pos_true = data['ligand'].xyz[data['ligand'].batch==idx].cpu().numpy().astype(np.float64) # + pocket_centers[idx]
        start_time = time.perf_counter()
        ff_corrected_mol, uncorrected_mol = correct_one(mol, pos_pred, method='ff')
        ff_time = time.perf_counter()
        align_corrected_mol, uncorrected_mol = correct_one(mol, pos_pred, method='align')
        aligned_time = time.perf_counter()
        poses.append([ff_corrected_mol.GetConformer().GetPositions(), align_corrected_mol.GetConformer().GetPositions(), pos_true])
        if len(mask) != 0:
            if mask[idx]:
                continue
        if out_init:
            # random position
            pos_init = data['ligand'].pos[data['ligand'].batch==idx].cpu().numpy().astype(np.float64) # + pocket_centers[idx]
            random_mol = copy.deepcopy(mol)
            random_mol = set_rdkit_mol_position(random_mol, pos_init)
            # random_file = f'{out_dir}/{data.pdb_id[idx]}/{data.pdb_id[idx]}_random_pos.sdf'
            random_file = f'{out_dir}/{data.pdb_id[idx]}_random_pos.sdf'
            if addHs:
                random_mol = Chem.AddHs(random_mol, addCoords=True)
            Chem.MolToMolFile(random_mol, random_file)
        if out_movie:
            # make movie
            pos_seq = data.pos_seq[:, data['ligand'].batch==idx, :].cpu().numpy().astype(np.float64)
            # movie_file = f'{out_dir}/{data.pdb_id[idx]}/{data.pdb_id[idx]}_pred_movie.sdf'
            movie_file = f'{out_dir}/{data.pdb_id[idx]}_pred_movie.sdf'
            make_movide(mol, pos_seq, movie_file)
        if out_corrected:
            ff_corrected_file = f'{out_dir}/{data.pdb_id[idx]}_pred_ff_corrected.sdf'
            try:
                if addHs:
                    ff_corrected_mol = Chem.AddHs(ff_corrected_mol, addCoords=True)
                Chem.MolToMolFile(ff_corrected_mol, ff_corrected_file)
            except:
                print(f'save {ff_corrected_file} failed')
                pass
            align_corrected_file = f'{out_dir}/{data.pdb_id[idx]}_pred_align_corrected.sdf'
            try:
                if addHs:
                    align_corrected_mol = Chem.AddHs(align_corrected_mol, addCoords=True)
                Chem.MolToMolFile(align_corrected_mol, align_corrected_file)
            except:
                print(f'save {ff_corrected_file} failed')
                pass
        if out_uncoorected:
            # uncorrected_file = f'{out_dir}/{data.pdb_id[idx]}/{data.pdb_id[idx]}_pred_uncorrected.sdf'
            uncorrected_file = f'{out_dir}/{data.pdb_id[idx]}_pred_uncorrected.sdf'
            if addHs:    
                uncorrected_mol = Chem.AddHs(uncorrected_mol, addCoords=True)
            Chem.MolToMolFile(uncorrected_mol, uncorrected_file)
    return poses, ff_time - start_time, aligned_time - ff_time
        

def make_movide(mol, pos_seq, movie_file):
    # pos_seq: numpy.array  shape = (25, atom_num, 3)
    with Chem.SDWriter(movie_file) as w:
        for i in range(pos_seq.shape[0]):
            pos_i = pos_seq[i]
            mol_i = copy.deepcopy(mol)
            mol_i = set_rdkit_mol_position(mol_i, pos_i)
            w.write(mol_i)


def correct_one(mol, pos_pred, method='ff'):
    # set pos
    raw_mol = copy.deepcopy(mol)
    pred_mol = copy.deepcopy(mol)
    pred_mol = set_rdkit_mol_position(pred_mol, pos_pred)
    # FF
    if method == 'ff':
        raw_mol = set_rdkit_mol_position(raw_mol, pos_pred)
        try:
            AllChem.MMFFOptimizeMolecule(raw_mol, maxIters=10)
        except:
            print('FF optimization failed')
    else:
        # get torsion_bonds
        rotable_bonds = get_torsions([pred_mol])
        # torsional align
        raw_mol = torsional_align(rdkit_mol=raw_mol, pred_conf=pred_mol.GetConformer(), rotable_bonds=rotable_bonds)
        # postion align
        # position_align_mol(rdkit_mol=mol, refer_mol=pred_mol)
        position_align_np(rdkit_mol=raw_mol, refer_mol=pred_mol)
    return raw_mol, pred_mol

def ff_complex_minization(pocket_mol, ligand_mol, n_iters=200, ff_type='mmff'):
    # form complex
    ligand_mol = AllChem.AddHs(ligand_mol, addCoords=True)
    complex_mol = Chem.CombineMols(pocket_mol, ligand_mol)
    try:
        Chem.SanitizeMol(complex_mol)
    except Chem.AtomValenceException:
        print('Invalid valence')
    except (Chem.AtomKekulizeException, Chem.KekulizeException):
        print('Failed to kekulize')
    try:
        if ff_type == 'mmff':
            ff = AllChem.MMFFGetMoleculeForceField(complex_mol, AllChem.MMFFGetMoleculeProperties(complex_mol), confId=0, ignoreInterfragInteractions=False)
        else:
            ff = AllChem.UFFGetMoleculeForceField(complex_mol, confId=0, ignoreInterfragInteractions=False)
        ff.Initialize()
        # fix pocket points
        [ff.AddFixedPoint(i) for i in range(pocket_mol.GetNumAtoms())]
        # minimize
        ff.Minimize(maxIts=n_iters)
        print(f"Performed {ff_type} with binding site...")
    except:
        print(f'Skip {ff_type}_{n_iters} ...')
    coords = complex_mol.GetConformer().GetPositions()
    rd_conf = ligand_mol.GetConformer()
    [rd_conf.SetAtomPosition(i, xyz) for i, xyz in enumerate(coords[-ligand_mol.GetNumAtoms():])] 
    return ligand_mol
       
    
    
if __name__ == '__main__':
    pass