#!usr/bin/env python3
# -*- coding:utf-8 -*-
# @time : 2022/3/28 14:08
# @author : Xujun Zhang
import copy
import glob
import os
from random import random
import sys
import MDAnalysis as mda
from functools import partial 
from multiprocessing import Pool
from copy import deepcopy
import numpy as np
import torch
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolAlign import RandomTransform
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

RDLogger.DisableLog("rdApp.*")
# dir of current
from utils.fns import load_graph, save_graph
from dataset.protein_feature import get_protein_feature_mda
from dataset.ligand_feature import get_ligand_feature_v1
from utils.post_processing import mmff_func


print = partial(print, flush=True)
   

class PDBBindGraphDataset(Dataset):

    def __init__(self, src_dir, pdb_ids, dst_dir, pki_labels=None, dataset_type='train', n_job=1,
                 on_the_fly=False,
                 verbose=False):
        '''

        :param src_dir: path for saving pocket file and ligand file
        :param pdb_ids: pdb id of protein file
        :param dst_dir: path for saving graph file
        :param pki_labels: pki/pkd/ic50 of protein-ligand complexes
        :param pocket_centers: the center of pocket (the center of the crystal ligand), (Num of complex, 3) np.array
        :param dataset_type: in ['train', 'valid', 'test']
        :param n_job: if n_job == 1: use for-loop;else: use multiprocessing
        :param on_the_fly: whether to get graph from a totoal graph list or a single graph file
        _______________________________________________________________________________________________________
        |  mode  |  generate single graph file  |  generate integrated graph file  |  load to memory at once  |
        |  False |          No                  |              Yes                 |            Yes           |
        |  True  |          Yes                 |              No                  |            No            |
        |  Fake  |          Yes                 |              No                  |            Yes           |
        _______________________________________________________________________________________________________
        '''
        self.src_dir = src_dir
        self.pdb_ids = pdb_ids
        self.dst_dir = dst_dir
        if pki_labels is not None:
            self.pki_labels = pki_labels
        else:
            self.pki_labels = np.zeros((len(self.pdb_ids)))
        os.makedirs(dst_dir, exist_ok=True)
        assert dataset_type in ['train', 'valid', 'test'], 'illegal dataset type'
        self.dataset_type = dataset_type
        self.dst_file = f'{dst_dir}/{dataset_type}.dgl'
        self.n_job = n_job
        assert on_the_fly in [True, False, 'Fake']
        self.verbose = verbose
        self.on_the_fly = on_the_fly
        self.graph_labels = []
        self.pre_process()

    def pre_process(self):
        if self.on_the_fly == 'Fake':
            self._generate_graph_on_the_fly_fake()
        elif self.on_the_fly:
            self._generate_graph_on_the_fly()
        else:
            self._generate_graph()

    def _generate_graph(self):
        if os.path.exists(self.dst_file):
            if self.verbose:
                print('load graph')
            self.graph_labels = load_graph(self.dst_file)
        else:
            idxs = range(len(self.pdb_ids))
            if self.verbose:
                print('### cal graph')
            single_process = partial(self._single_process, return_graph=True, save_file=False)
            # generate graph
            if self.n_job == 1:
                if self.verbose:
                    idxs = tqdm(idxs)
                for idx in idxs:
                    self.graph_labels.append(single_process(idx))
            else:
                pool = Pool(self.n_job)
                self.graph_labels = pool.map(single_process, idxs)
                pool.close()
                pool.join()
            # filter None
            self.graph_labels = list(filter(lambda x: x is not None, self.graph_labels))
            # save file
            save_graph(self.dst_file, self.graph_labels)

    def _generate_graph_on_the_fly(self):
        idxs = range(len(self.pdb_ids))
        if self.verbose:
            print('### get graph on the fly')
        single_process = partial(self._single_process, return_graph=False, save_file=True)
        # generate graph
        if self.n_job == 1:
            if self.verbose:
                idxs = tqdm(idxs)
            for idx in idxs:
                single_process(idx)
        else:
            pool = Pool(self.n_job)
            pool.map(single_process, idxs)
            pool.close()
            pool.join()
        # self.pdb_ids = [os.path.split(i)[-1].split('.')[0] for i in glob.glob(f'{self.dst_dir}/*.dgl')]

    def _generate_graph_on_the_fly_fake(self):
        idxs = range(len(self.pdb_ids))
        if self.verbose:
            print('### get graph on the fly (fake)')
        single_process = partial(self._single_process, return_graph=True, save_file=True)
        # generate graph
        if self.n_job == 1:
            if self.verbose:
                idxs = tqdm(idxs)
            for idx in idxs:
                self.graph_labels.append(single_process(idx))
        else:
            pool = Pool(self.n_job)
            self.graph_labels = pool.map(single_process, idxs)
            pool.close()
            pool.join()
        # filter None
        self.graph_labels = list(filter(lambda x: x is not None, self.graph_labels))

    def _single_process(self, idx, return_graph=False, save_file=False):
        pdb_id = self.pdb_ids[idx]
        dst_file = f'{self.dst_dir}/{pdb_id}.dgl'
        if os.path.exists(dst_file):
            # reload graph
            if return_graph:
                return load_graph(dst_file)
        else:
            # generate graph
            pki_label = self.pki_labels[idx]
            src_path_local = f'{self.src_dir}/{pdb_id}'
            pocket_pdb = f'{src_path_local}/{pdb_id}_pocket_ligH12A.pdb'
            ligand_crystal_mol2 = f'{src_path_local}/{pdb_id}_ligand.mol2'
            ligand_crystal_sdf = f'{src_path_local}/{pdb_id}_ligand.sdf'
            try:
                data = get_graph_v1(pocket_pdb=pocket_pdb,
                                        ligand_crystal_mol2=ligand_crystal_mol2,
                                        ligand_crystal_sdf=ligand_crystal_sdf)
                data.pdb_id = pdb_id
                if save_file:
                    save_graph(dst_file, data)
                if return_graph:
                    return data
            except:
                print(f'{pdb_id} error')
                return None

    def __getitem__(self, idx):
        if self.on_the_fly == True:
            data = self._single_process(idx=idx, return_graph=True, save_file=False)
        else:
            data = self.graph_labels[idx]
        data['ligand'].pos = random_rotation(shuffle_center(data['ligand'].pos))  
        return data


    def __len__(self):
        if self.on_the_fly == True:
            return len(self.pdb_ids)
        else:
            return len(self.graph_labels)



class MultiComplexGraphDataset_Fly(Dataset):

    def __init__(self, complex_path, protein_ligand_names, reload_g=False):
        self.complex_path = complex_path
        self.protein_ligand_names = protein_ligand_names
        self.protein2data_dict = {}
        if reload_g:
            self.process_fn = self._reload_process
        else:
            self.process_fn = self._single_process
    
    def preprocessing_fn(self, idx):
        try:
            self.process_fn(idx)
        except:
            print(f'{"_".join(self.protein_ligand_names[idx])} error')
    
    def preprocessing(self, n_job=-1):
        if n_job == 1:
            for idx in range(len(self.protein_ligand_names)):
                self.preprocessing_fn(idx)
        else:
            pool = Pool()
            pool.map(self.preprocessing_fn, range(len(self.protein_ligand_names)))
            pool.close()
            pool.join()
    
    def _single_process(self, idx):
        # torch.set_num_threads(1)
        protein_name, ligand_name = self.protein_ligand_names[idx]
        complex_local_path = f'{self.complex_path}/{protein_name}'
        # get protein_graph
        if protein_name not in self.protein2data_dict.keys():
            pocket_pdb = f'{complex_local_path}/{protein_name}_pocket_ligH12A.pdb'
            # get pocket_center
            cry_ligand_mol2 = f'{complex_local_path}/{protein_name}_ligand.mol2'
            pocket_center = torch.tensor(get_pocker_center_from_cmd(cry_ligand_mol2), dtype=torch.float32)
            # generate graph
            protein_graph = generate_protein_graph(pocket_pdb)
            protein_graph.pocket_center=pocket_center
            self.protein2data_dict[protein_name] = protein_graph
        else:
            protein_graph = self.protein2data_dict[protein_name]
        # generate graph
        ligand_sdf = f'{complex_local_path}/ligand/{ligand_name}.sdf'
        ligand_mol = file2conformer(ligand_sdf)
        l_xyz = ligand_mol.GetConformer().GetPositions()
        ligand_mol = mol2conformer_v1(ligand_mol)
        data = get_graph_v2(protein_graph, cry_ligand_mol=ligand_mol)
        data.pdb_id = f'{protein_name}_{ligand_name}'
        data['ligand'].pos = data['ligand'].xyz
        data['ligand'].xyz = torch.from_numpy(l_xyz).to(torch.float32)
        return data

    def _reload_process(self, idx):
        # torch.set_num_threads(1)
        protein_name, ligand_name = self.protein_ligand_names[idx]
        complex_local_path = f'{self.complex_path}/{protein_name}'
        prorein_graph_file = f'{complex_local_path}/{protein_name}.dgl'
        # get protein_graph
        if not os.path.exists(prorein_graph_file):
            pocket_pdb = f'{complex_local_path}/{protein_name}_pocket_ligH12A.pdb'
            # get pocket_center
            cry_ligand_mol2 = f'{complex_local_path}/{protein_name}_ligand.mol2'
            pocket_center = torch.tensor(get_pocker_center_from_cmd(cry_ligand_mol2), dtype=torch.float32)
            # generate graph
            protein_data = generate_protein_graph(pocket_pdb)
            protein_data.pocket_center=pocket_center
            save_graph(prorein_graph_file, protein_data)
        else:
            protein_data = load_graph(prorein_graph_file)
        # generate graph
        ligand_sdf = f'{complex_local_path}/ligand/{ligand_name}.sdf'
        ligand_graph_file = f'{complex_local_path}/graph/{ligand_name}.dgl'
        if not os.path.exists(ligand_graph_file):
            os.makedirs(os.path.split(ligand_graph_file)[0], exist_ok=True)
            ligand_mol = file2conformer(ligand_sdf)
            l_xyz = ligand_mol.GetConformer().GetPositions()
            ligand_mol = mol2conformer_v1(ligand_mol)
            data = HeteroData()
            ligand_data = generate_lig_graph(data, ligand_mol)
            ligand_data['ligand'].pos = ligand_data['ligand'].xyz
            ligand_data['ligand'].xyz = torch.from_numpy(l_xyz).to(torch.float32)
            save_graph(ligand_graph_file, ligand_data)
        else:
            ligand_data = load_graph(ligand_graph_file)
        data = merge_pro_lig_graph(pro_data=protein_data, data=ligand_data)
        data = get_protein_ligand_graph(data, pro_node_num=data['protein'].xyz.size(0), lig_node_num=data['ligand'].xyz.size(0))
        data.pdb_id = f'{protein_name}_{ligand_name}'
        data.pocket_center = protein_data.pocket_center
        return data


    def __getitem__(self, idx):
        try:
            data = self.process_fn(idx)
            data['ligand'].pos = random_rotation(shuffle_center(move2center(data['ligand'].pos, pocket_center=data.pocket_center)))  
        except:
            # pass
            print(f'{"_".join(self.protein_ligand_names[idx])} error')
            return None
        return data

    def __len__(self):
        return len(self.protein_ligand_names)


class VSTestGraphDataset_Fly(Dataset):

    def __init__(self, protein_file, ligand_path, pocket_center):
        self.ligand_names = []
        self.ligand_smis = []
        self.protein_file = protein_file
        self.ligand_path = ligand_path
        torch.set_num_threads(1)
        self.protein_data = generate_protein_graph(pocket_pdb=self.protein_file)
        self.pocket_center = pocket_center
        self.protein_data.pocket_center = pocket_center

    
    def _get_mol(self, idx):
        return None
    
    def _single_process(self, idx):
        # torch.set_num_threads(1)
        ligand_name = self.ligand_names[idx]
        # generate graph
        cry_ligand_mol = self._get_mol(idx)
        data = get_graph_v2(self.protein_data, cry_ligand_mol=cry_ligand_mol)
        data.pdb_id = ligand_name
        return data


    def __getitem__(self, idx):
        try:
            data = self._single_process(idx)
            data['ligand'].pos = random_rotation(shuffle_center(data['ligand'].pos))  
        except:
            # pass
            # print(f'{self.ligand_names[idx]} error')
            return None
        return data

    def __len__(self):
        return len(self.ligand_names)
    
class VSTestGraphDataset_Fly_SMI(VSTestGraphDataset_Fly):
    '''initializing the ligand pose with rdkit with mols from SMILES'''
    def __init__(self, protein_file, ligand_path, pocket_center):
        super().__init__(protein_file, ligand_path, pocket_center)
        self.ligand_smis = []
        with open(ligand_path, 'r') as f:
            con = f.read().splitlines()
        self.ligand_names = [i.split()[1] for i in con] #  if i.split()[1]=='BDB14479']
        self.ligand_smis = [i.split()[0] for i in con] #  if i.split()[1]=='BDB14479']

    def _get_mol(self, idx):
        smi = self.ligand_smis[idx]
        mol = smi2conformer(smi)
        # mol = smi2conformer_fast(smi)
        return mol
    
    def _single_process(self, idx):
        # torch.set_num_threads(1)
        ligand_name = self.ligand_names[idx]
        # generate graph
        cry_ligand_mol = self._get_mol(idx)
        data = get_graph_v2(self.protein_data.clone(), cry_ligand_mol=cry_ligand_mol)
        data.pdb_id = ligand_name
        data['ligand'].mol = cry_ligand_mol
        data['ligand'].pos = (data['ligand'].xyz + data.pocket_center - data['ligand'].xyz.mean(dim=0)).to(torch.float32)
        return data
    
class VSTestGraphDataset_FlyReload_SMI(VSTestGraphDataset_Fly):
    '''initializing the ligand pose with rdkit with mols from SMILES'''
    def __init__(self, protein_file, ligand_path, pocket_center):
        super().__init__(protein_file, ligand_path, pocket_center)
        if not os.path.exists(ligand_path):
            os.makedirs(ligand_path, exist_ok=True)
            self.ligand_names = []
        else:
            self.ligand_names = [ligand_file.split('.')[0] for ligand_file in os.listdir(ligand_path)]
        self.graph_dir = ligand_path

    def generate_graphs(self, ligand_smi, n_job=-1, verbose=True):
        with open(ligand_smi, 'r') as f:
            con = f.read().splitlines()
        self.ligand_names = [i.split()[1] for i in con] #  if i.split()[1]=='BDB14479']
        self.ligand_smis = [i.split()[0] for i in con] #  if i.split()[1]=='BDB14479']
        if n_job == 1:
            if verbose:
                iters = tqdm(range(len(self.ligand_names)))
            else:
                iters = range(len(self.ligand_names))            
            for idx in iters:
                self._single_process(idx)
        else:
            pool = Pool()
            pool.map(self._single_process, range(len(self.ligand_names)))
            pool.close()
            pool.join()
        print('reinitialize')
        self.ligand_names = [ligand_file.split('.')[0] for ligand_file in os.listdir(self.graph_dir)]
        
    def _get_mol(self, idx):
        smi = self.ligand_smis[idx]
        mol = smi2conformer(smi)
        # mol = smi2conformer_fast(smi)
        return mol
    
    def _single_process(self, idx):
        torch.set_num_threads(1)
        ligand_name = self.ligand_names[idx]
        dst_file = f"{self.graph_dir}/{ligand_name.replace('/', '_')}.dgl"
        # generate graph
        try:
            ligand_mol = self._get_mol(idx)
            data = HeteroData()
            ligand_data = generate_lig_graph(data, ligand_mol)
            ligand_data.pdb_id = ligand_name
            ligand_data['ligand'].mol = ligand_mol
            ligand_data['ligand'].pos = (ligand_data['ligand'].xyz + self.pocket_center - ligand_data['ligand'].xyz.mean(dim=0)).to(torch.float32)
            save_graph(dst_file, ligand_data)
        except:
            None
            
    def merge_complex_graph(self, idx):
        ligand_name = self.ligand_names[idx]
        dst_file = f'{self.graph_dir}/{ligand_name}.dgl'
        ligand_data = load_graph(dst_file)
        data = merge_pro_lig_graph(pro_data=self.protein_data.clone(), data=ligand_data)
        data = get_protein_ligand_graph(data, pro_node_num=data['protein'].xyz.size(0), lig_node_num=data['ligand'].xyz.size(0))
        return data
    
    def __getitem__(self, idx):
        try:
            data = self.merge_complex_graph(idx)
            data['ligand'].pos = random_rotation(shuffle_center(data['ligand'].pos))  
        except:
            # pass
            # print(f'{self.ligand_names[idx]} error')
            return None
        return data
    
    
class VSTestGraphDataset_Fly_SDFMOL2_Refined(VSTestGraphDataset_Fly):
    '''refined the ligand conformation initialized with provied pose from SDF/MOL2 files'''
    def __init__(self, protein_file, ligand_path, pocket_center):
        super().__init__(protein_file, ligand_path, pocket_center)
        self.ligand_names = list(set([i.split('_')[0] for i in os.listdir(ligand_path)]))

    def _get_mol(self, idx):
        ligand_name = self.ligand_names[idx]
        lig_file_sdf = f'{self.ligand_path}/{ligand_name}_pred_uncorrected.sdf'
        lig_file_mol2 = f'{self.ligand_path}/{ligand_name}.mol2'
        mol = file2conformer(lig_file_sdf, lig_file_mol2)
        return mol, ligand_name
    
    
    def _single_process(self, idx):
        torch.set_num_threads(1)
        # generate graph
        cry_ligand_mol, ligand_name = self._get_mol(idx)
        data = get_graph_v2(self.protein_data.clone(), cry_ligand_mol=cry_ligand_mol)
        data.pdb_id = ligand_name
        data['ligand'].mol = cry_ligand_mol
        data['ligand'].pos = data['ligand'].xyz 
        return data
    
    def __getitem__(self, idx):
        try:
            data = self._single_process(idx)
        except:
            return None
        return data
    
class VSTestGraphDataset_Fly_SDFMOL2(VSTestGraphDataset_Fly_SDFMOL2_Refined):
    '''generating the ligand conformation initialized by rdkit EDGKT with mols from SDF/MOL2 files'''
    def __init__(self, protein_file, ligand_path, pocket_center, geometric_pos_init=True, use_rdkit_pos=True):
        super().__init__(protein_file, ligand_path, pocket_center, geometric_pos_init, use_rdkit_pos)
        self.ligand_names = [i.split('.')[0] for i in os.listdir(ligand_path)]
    
    def _single_process(self, idx):
        # generate graph
        cry_ligand_mol, ligand_name = self._get_mol(idx)
        l_xyz = cry_ligand_mol.GetConformer().GetPositions()
        ### different  
        cry_ligand_mol = mol2conformer_v1(cry_ligand_mol)
        ###
        data = get_graph_v2(self.protein_data.clone(), cry_ligand_mol=cry_ligand_mol)
        data.pdb_id = ligand_name
        data['ligand'].pos = data['ligand'].xyz + data.pocket_center - data['ligand'].xyz.mean(dim=0)
        data['ligand'].xyz = torch.from_numpy(l_xyz).to(torch.float32)
        return data
    

def get_repeat_node(src_num, dst_num):
    return torch.arange(src_num, dtype=torch.long).repeat(dst_num), \
           torch.as_tensor(np.repeat(np.arange(dst_num), src_num), dtype=torch.long)


def generate_graph_4_Multi_PL(pocket_mol, ligand_mol, use_rdkit_pos=True):
    # get pocket
    l_xyz =  torch.from_numpy(ligand_mol.GetConformer().GetPositions()).to(torch.float32)
    pocket_center = l_xyz.mean(dim=0)
    # get rdkit pos
    if use_rdkit_pos:
        rdkit_mol = mol2conformer_v1(ligand_mol)
    else:
        rdkit_mol = ligand_mol
    # get feats
    p_xyz, p_xyz_full, p_seq, p_node_s, p_node_v, p_edge_index, p_edge_s, p_edge_v, p_full_edge_s = get_protein_feature_mda(pocket_mol)
    l_xyz_rdkit, l_node_feature, l_edge_index, l_edge_feature, l_full_edge_s, l_interaction_edge_mask, l_cov_edge_mask = get_ligand_feature_v1(
        rdkit_mol)
    # to data
    data = HeteroData()
    # protein
    data.pocket_center = pocket_center.view((1, 3)).to(torch.float32)
    data['protein'].node_s = p_node_s.to(torch.float32) 
    data['protein'].node_v = p_node_v.to(torch.float32)
    data['protein'].xyz = p_xyz.to(torch.float32) 
    data['protein'].xyz_full = p_xyz_full.to(torch.float32) 
    data['protein'].seq = p_seq.to(torch.int32)
    data['protein', 'p2p', 'protein'].edge_index = p_edge_index.to(torch.long)
    data['protein', 'p2p', 'protein'].edge_s = p_edge_s.to(torch.float32) 
    data['protein', 'p2p', 'protein'].full_edge_s = p_full_edge_s.to(torch.float32) 
    data['protein', 'p2p', 'protein'].edge_v = p_edge_v.to(torch.float32) 
    # ligand
    data['ligand'].xyz = l_xyz.to(torch.float32)
    data['ligand'].node_s = l_node_feature.to(torch.int32)
    data['ligand'].interaction_edge_mask = l_interaction_edge_mask
    data['ligand'].cov_edge_mask = l_cov_edge_mask
    data['ligand', 'l2l', 'ligand'].edge_index = l_edge_index.to(torch.long)
    data['ligand', 'l2l', 'ligand'].edge_s = l_edge_feature.to(torch.int32)
    data['ligand', 'l2l', 'ligand'].full_edge_s = l_full_edge_s.to(torch.float32)
    # sym_index
    data.sym_index = get_sym_index(ligand_mol, ligand_mol)
    data['ligand'].mol = rdkit_mol
    data['ligand'].pos = l_xyz_rdkit.to(torch.float32)
    # move ligand to the pocket_center
    data['ligand'].pos -= (data['ligand'].pos.mean(axis=0) - pocket_center)
    # protein-ligand
    data['protein', 'p2l', 'ligand'].edge_index = torch.stack(
        get_repeat_node(p_xyz.shape[0], l_xyz.shape[0]), dim=0)
    return data

def generate_protein_graph(pocket_pdb):
    pocket_mol = mda.Universe(pocket_pdb)
    # get feats
    p_xyz, p_xyz_full, p_seq, p_node_s, p_node_v, p_edge_index, p_edge_s, p_edge_v, p_full_edge_s = get_protein_feature_mda(pocket_mol)
    # to data
    data = HeteroData()
    # protein
    data['protein'].node_s = p_node_s.to(torch.float32) 
    data['protein'].node_v = p_node_v.to(torch.float32)
    data['protein'].xyz = p_xyz.to(torch.float32) 
    data['protein'].xyz_full = p_xyz_full.to(torch.float32) 
    data['protein'].seq = p_seq.to(torch.int32)
    data['protein', 'p2p', 'protein'].edge_index = p_edge_index.to(torch.long)
    data['protein', 'p2p', 'protein'].edge_s = p_edge_s.to(torch.float32) 
    data['protein', 'p2p', 'protein'].full_edge_s = p_full_edge_s.to(torch.float32) 
    data['protein', 'p2p', 'protein'].edge_v = p_edge_v.to(torch.float32) 
    return data


def generate_lig_graph(data, ligand_mol):
    # get feats
    l_xyz, l_node_feature, l_edge_index, l_edge_feature, l_full_edge_s, l_interaction_edge_mask, l_cov_edge_mask = get_ligand_feature_v1(
    ligand_mol)
    # ligand
    data['ligand'].xyz = l_xyz.to(torch.float32)
    data['ligand'].node_s = l_node_feature.to(torch.int32)
    data['ligand'].cov_edge_mask = l_cov_edge_mask
    data['ligand', 'l2l', 'ligand'].edge_index = l_edge_index.to(torch.long)
    data['ligand', 'l2l', 'ligand'].edge_s = l_edge_feature.to(torch.int32)
    data['ligand', 'l2l', 'ligand'].full_edge_s = l_full_edge_s.to(torch.float32)
    return data

def get_protein_ligand_graph(data, pro_node_num, lig_node_num):
    # protein-ligand
    data['protein', 'p2l', 'ligand'].edge_index = torch.stack(
    get_repeat_node(pro_node_num, lig_node_num), dim=0)
    return data

def mol2conformer(mol, n_confs):
    mol_rdkit = copy.deepcopy(mol)
    # remove all the conformers
    mol_rdkit.RemoveAllConformers()
    # add H atom for more accurate conformation
    mol_rdkit = Chem.AddHs(mol_rdkit)
    # calculate conformation
    AllChem.EmbedMultipleConfs(mol_rdkit, numConfs=n_confs)
    # minimization conformation
    mmff_func(mol_rdkit)
    # remove H atom
    mol_rdkit = Chem.RemoveHs(mol_rdkit)
    return mol_rdkit

def ff_refined_mol_pos(mol, n_max=1):
    feed_back = [[-1, 1]]
    n = 0
    while feed_back[0][0] == -1 and n < n_max:
        feed_back = AllChem.MMFFOptimizeMoleculeConfs(mol, mmffVariant='MMFF94s')
        n += 1
    return mol

def add_conformer(mol):
    feed_back = AllChem.EmbedMolecule(mol)
    if feed_back == -1:
        return -1
    return mol

def mol2conformer_v1(mol):
    m_mol = copy.deepcopy(mol)
    m_mol = Chem.AddHs(m_mol)
    m_mol = add_conformer(m_mol)
    if m_mol == -1:
        return mol
    m_mol = ff_refined_mol_pos(m_mol)
    m_mol = Chem.RemoveAllHs(m_mol)
    return m_mol

def mol2conformer_v2(mol):
    m_mol = copy.deepcopy(mol)
    AllChem.Compute2DCoords(m_mol)
    return m_mol


def smi2conformer(smi):
    mol = Chem.MolFromSmiles(Chem.MolToSmiles(Chem.AddHs(Chem.MolFromSmiles(smi))))
    mol = Chem.AddHs(mol)
    smi = Chem.MolToSmiles(mol)
    mol = Chem.MolFromSmiles(smi)
    mol = Chem.AddHs(mol)
    m_mol = add_conformer(mol)
    if m_mol != -1:
        mol = m_mol
    mol = ff_refined_mol_pos(mol, n_max=10000)
    mol = Chem.RemoveAllHs(mol)
    return mol


def smi2conformer_fast(smi):
    mol = Chem.MolFromSmiles(smi)
    mol = Chem.AddHs(mol)
    m_mol = add_conformer(mol)
    if m_mol != -1:
        mol = m_mol
    mol = ff_refined_mol_pos(mol, n_max=1)
    mol = Chem.RemoveAllHs(mol)
    return mol


def file2conformer(*args):
    for f in args:
        try:
            if os.path.splitext(f)[-1] == '.sdf':
                mol = Chem.MolFromMolFile(f, removeHs=True)
            else:
                mol = Chem.MolFromMol2File(f, removeHs=True)
            if mol is not None:
                mol = Chem.RemoveAllHs(mol)
                return mol
        except:
            continue


def get_graph_v1(pocket_pdb, ligand_smi='', ligand_crystal_mol2='', ligand_crystal_sdf='', pocket_center=np.array([])):
    torch.set_num_threads(1)
    # get protein mol
    pocket_mol = mda.Universe(pocket_pdb)
    # get ligand_mol
    cry_ligand_mol = file2conformer(ligand_crystal_sdf, ligand_crystal_mol2)
    # generate graph
    hg = generate_graph_4_Multi_PL(pocket_mol, cry_ligand_mol, use_rdkit_pos=True)
    return hg

def get_graph_v2(pro_data, cry_ligand_mol):
    # generate graph
    data = generate_lig_graph(pro_data, cry_ligand_mol)
    # pro_lig
    data = get_protein_ligand_graph(data, pro_node_num=data['protein'].xyz.size(0), lig_node_num=data['ligand'].xyz.size(0))
    return data

def merge_pro_lig_graph(pro_data, data):
    # pro
    data['protein'].node_s = pro_data['protein'].node_s
    data['protein'].node_v = pro_data['protein'].node_v 
    data['protein'].xyz = pro_data['protein'].xyz 
    data['protein'].xyz_full = pro_data['protein'].xyz_full 
    data['protein'].seq = pro_data['protein'].seq 
    data['protein', 'p2p', 'protein'].edge_index = pro_data['protein', 'p2p', 'protein'].edge_index 
    data['protein', 'p2p', 'protein'].edge_s = pro_data['protein', 'p2p', 'protein'].edge_s 
    data['protein', 'p2p', 'protein'].full_edge_s = pro_data['protein', 'p2p', 'protein'].full_edge_s 
    data['protein', 'p2p', 'protein'].edge_v = pro_data['protein', 'p2p', 'protein'].edge_v 
    return data

def get_graph_pro(pocket_pdb, pocket_center=np.array([])):
    torch.set_num_threads(1)
    # get protein mol
    # parser = PDBParser(QUIET=True)
    # pocket_mol = parser.get_structure("x", pocket_pdb)
    pocket_mol = mda.Universe(pocket_pdb)
    # generate graph
    pg = generate_protein_graph(pocket_mol, pocket_center)
    return pg

def get_sym_index(prb_mol, ref_mol):
    try:
        sym_index = np.asarray(prb_mol.GetSubstructMatches(ref_mol, uniquify=False))
    except:
        sym_index = np.arange(prb_mol.GetNumAtoms()).reshape((1, -1))
    if len(sym_index) == 0:
        sym_index = np.arange(prb_mol.GetNumAtoms()).reshape((1, -1))
    return sym_index

def RandomRotatePos(mol):
    RandomTransform(mol)
    pos = mol.GetConformer().GetPositions()
    pos = pos - pos.mean(axis=0)
    return pos


def move2center(xyz, pocket_center):
    xyz += pocket_center - xyz.mean(axis=0)
    return xyz

def shuffle_center(xyz, noise=4):
    return xyz + torch.normal(mean=0, std=noise, size=(1, 3), dtype=torch.float) 
    
def random_rotation(xyz):
    random_rotation_matrix = torch.from_numpy(R.random().as_matrix()).to(torch.float32)
    lig_center = xyz.mean(dim=0)
    return (xyz - lig_center)@random_rotation_matrix.T + lig_center
    
def get_pocker_center_from_cmd(ligand_mol2):
    x = os.popen(
        "cat %s | sed -n '/@<TRIPOS>ATOM/,/@<TRIPOS>BOND/'p | awk '{print $3}' | awk '{x+=$1} END {print x/(NR-2)}'" % ligand_mol2).read()
    y = os.popen(
        "cat %s | sed -n '/@<TRIPOS>ATOM/,/@<TRIPOS>BOND/'p | awk '{print $4}' | awk '{y+=$1} END {print y/(NR-2)}'" % ligand_mol2).read()
    z = os.popen(
        "cat %s | sed -n '/@<TRIPOS>ATOM/,/@<TRIPOS>BOND/'p | awk '{print $5}' | awk '{z+=$1} END {print z/(NR-2)}'" % ligand_mol2).read()
    return float(x), float(y), float(z)


def get_mol2_xyz_from_cmd(ligand_mol2):
    x = os.popen(
        "cat %s | sed -n '/@<TRIPOS>ATOM/,/@<TRIPOS>BOND/'p | awk '{print $3}'" % ligand_mol2).read().splitlines()[1:-1]
    y = os.popen(
        "cat %s | sed -n '/@<TRIPOS>ATOM/,/@<TRIPOS>BOND/'p | awk '{print $4}'" % ligand_mol2).read().splitlines()[1:-1]
    z = os.popen(
        "cat %s | sed -n '/@<TRIPOS>ATOM/,/@<TRIPOS>BOND/'p | awk '{print $5}'" % ligand_mol2).read().splitlines()[1:-1]
    return np.asanyarray(list(zip(x, y, z))).astype(float)


if __name__ == '__main__':
    lig_mol2 = '/root/project_7/data/pretrain_pdbbind/3vjs/3vjs_ligand.mol2'
    get_mol2_xyz_from_cmd(lig_mol2)
