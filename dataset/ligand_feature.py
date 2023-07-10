#!usr/bin/env python3
# -*- coding:utf-8 -*-
# @time : 2022/8/8 14:33
# @author : Xujun Zhang

from copy import deepcopy
import torch
import networkx as nx
import numpy as np
import copy
from torch_geometric.utils import to_networkx
from rdkit import Chem
import warnings
from rdkit.Chem import AllChem
from torch_geometric.utils import to_dense_adj, dense_to_sparse
warnings.filterwarnings("ignore")

def binarize(x):
    return torch.where(x > 0, torch.ones_like(x), torch.zeros_like(x))

def get_higher_order_adj_matrix(adj, order):
    """
    Args:
        adj:        (N, N)
        type_mat:   (N, N)
    Returns:
        Following attributes will be updated:
            - edge_index
            - edge_type
        Following attributes will be added to the data object:
            - bond_edge_index:  Original edge_index.
    """
    adj_mats = [torch.eye(adj.size(0), dtype=torch.long, device=adj.device), \
                binarize(adj + torch.eye(adj.size(0), dtype=torch.long, device=adj.device))]

    for i in range(2, order + 1):
        adj_mats.append(binarize(adj_mats[i - 1] @ adj_mats[1]))
    order_mat = torch.zeros_like(adj)

    for i in range(1, order + 1):
        order_mat += (adj_mats[i] - adj_mats[i - 1]) * i

    return order_mat

def get_ring_adj_matrix(mol, adj):
    new_adj = deepcopy(adj)
    ssr = Chem.GetSymmSSSR(mol)
    for ring in ssr:
        # print(ring)
        for i in ring:
            for j in ring:
                if i==j:
                    continue
                elif new_adj[i][j] != 1:
                    new_adj[i][j]+=1
    return new_adj




# orderd by perodic table
atom_vocab = ["H", "B", "C", "N", "O", "F", "Mg", "Si", "P", "S", "Cl", "Cu", "Zn", "Se", "Br", "Sn", "I"]
atom_vocab = {a: i for i, a in enumerate(atom_vocab)}
degree_vocab = range(7)
num_hs_vocab = range(7)
formal_charge_vocab = range(-5, 6)
chiral_tag_vocab = range(4)
total_valence_vocab = range(8)
num_radical_vocab = range(8)
hybridization_vocab = range(len(Chem.rdchem.HybridizationType.values))

bond_type_vocab = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
                   Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
bond_type_vocab = {b: i for i, b in enumerate(bond_type_vocab)}
bond_dir_vocab = range(len(Chem.rdchem.BondDir.values))
bond_stereo_vocab = range(len(Chem.rdchem.BondStereo.values))

# orderd by molecular mass
residue_vocab = ["GLY", "ALA", "SER", "PRO", "VAL", "THR", "CYS", "ILE", "LEU", "ASN",
                 "ASP", "GLN", "LYS", "GLU", "MET", "HIS", "PHE", "ARG", "TYR", "TRP"]


def onehot(x, vocab, allow_unknown=False):
    if x in vocab:
        if isinstance(vocab, dict):
            index = vocab[x]
        else:
            index = vocab.index(x)
    else:
        index = -1
    if allow_unknown:
        feature = [0] * (len(vocab) + 1)
        if index == -1:
            warnings.warn("Unknown value `%s`" % x)
        feature[index] = 1
    else:
        feature = [0] * len(vocab)
        if index == -1:
            raise ValueError("Unknown value `%s`. Available vocabulary is `%s`" % (x, vocab))
        feature[index] = 1

    return feature


def atom_default(atom):
    """Default atom feature.

    Features:
        GetSymbol(): one-hot embedding for the atomic symbol dim=18
        
        GetChiralTag(): one-hot embedding for atomic chiral tag dim=5
        
        GetTotalDegree(): one-hot embedding for the degree of the atom in the molecule including Hs dim=5
        
        GetFormalCharge(): one-hot embedding for the number of formal charges in the molecule
        
        GetTotalNumHs(): one-hot embedding for the total number of Hs (explicit and implicit) on the atom 
        
        GetNumRadicalElectrons(): one-hot embedding for the number of radical electrons on the atom
        
        GetHybridization(): one-hot embedding for the atom's hybridization
        
        GetIsAromatic(): whether the atom is aromatic
        
        IsInRing(): whether the atom is in a ring
        18 + 5 + 8 + 12 + 8 + 9 + 10 + 9 + 3 + 4 
    """
    return onehot(atom.GetSymbol(), atom_vocab, allow_unknown=True) + \
           onehot(atom.GetChiralTag(), chiral_tag_vocab, allow_unknown=True) + \
           onehot(atom.GetTotalDegree(), degree_vocab, allow_unknown=True) + \
           onehot(atom.GetFormalCharge(), formal_charge_vocab, allow_unknown=True) + \
           onehot(atom.GetTotalNumHs(), num_hs_vocab, allow_unknown=True) + \
           onehot(atom.GetNumRadicalElectrons(), num_radical_vocab, allow_unknown=True) + \
           onehot(atom.GetHybridization(), hybridization_vocab, allow_unknown=True) + \
            onehot(atom.GetTotalValence(), total_valence_vocab, allow_unknown=True) + \
           [atom.GetIsAromatic(), atom.IsInRing(), atom.IsInRing() and (not atom.IsInRingSize(3)) and (not atom.IsInRingSize(4)) \
            and (not atom.IsInRingSize(5)) and (not atom.IsInRingSize(6))]+[atom.IsInRingSize(i) for i in range(3, 7)]


def atom_center_identification(atom):
    """Reaction center identification atom feature.

    Features:
        GetSymbol(): one-hot embedding for the atomic symbol
        
        GetTotalNumHs(): one-hot embedding for the total number of Hs (explicit and implicit) on the atom 
        
        GetTotalDegree(): one-hot embedding for the degree of the atom in the molecule including Hs
        
        GetTotalValence(): one-hot embedding for the total valence (explicit + implicit) of the atom
        
        GetIsAromatic(): whether the atom is aromatic
        
        IsInRing(): whether the atom is in a ring
    """
    return onehot(atom.GetSymbol(), atom_vocab, allow_unknown=True) + \
           onehot(atom.GetTotalNumHs(), num_hs_vocab) + \
           onehot(atom.GetTotalDegree(), degree_vocab, allow_unknown=True) + \
           onehot(atom.GetTotalValence(), total_valence_vocab) + \
           [atom.GetIsAromatic(), atom.IsInRing()]



def atom_synthon_completion(atom):
    """Synthon completion atom feature.

    Features:
        GetSymbol(): one-hot embedding for the atomic symbol

        GetTotalNumHs(): one-hot embedding for the total number of Hs (explicit and implicit) on the atom 
        
        GetTotalDegree(): one-hot embedding for the degree of the atom in the molecule including Hs
        
        IsInRing(): whether the atom is in a ring
        
        IsInRingSize(3, 4, 5, 6): whether the atom is in a ring of a particular size
        
        IsInRing() and not IsInRingSize(3, 4, 5, 6): whether the atom is in a ring and not in a ring of 3, 4, 5, 6
    """
    return onehot(atom.GetSymbol(), atom_vocab, allow_unknown=True) + \
           onehot(atom.GetTotalNumHs(), num_hs_vocab) + \
           onehot(atom.GetTotalDegree(), degree_vocab, allow_unknown=True) + \
           [atom.IsInRing(), atom.IsInRingSize(3), atom.IsInRingSize(4),
            atom.IsInRingSize(5), atom.IsInRingSize(6), 
            atom.IsInRing() and (not atom.IsInRingSize(3)) and (not atom.IsInRingSize(4)) \
            and (not atom.IsInRingSize(5)) and (not atom.IsInRingSize(6))]



def atom_symbol(atom):
    """Symbol atom feature.

    Features:
        GetSymbol(): one-hot embedding for the atomic symbol
    """
    return onehot(atom.GetSymbol(), atom_vocab, allow_unknown=True)



def atom_explicit_property_prediction(atom):
    """Explicit property prediction atom feature.

    Features:
        GetSymbol(): one-hot embedding for the atomic symbol

        GetDegree(): one-hot embedding for the degree of the atom in the molecule

        GetTotalValence(): one-hot embedding for the total valence (explicit + implicit) of the atom
        
        GetFormalCharge(): one-hot embedding for the number of formal charges in the molecule
        
        GetIsAromatic(): whether the atom is aromatic
    """
    return onehot(atom.GetSymbol(), atom_vocab, allow_unknown=True) + \
           onehot(atom.GetDegree(), degree_vocab, allow_unknown=True) + \
           onehot(atom.GetTotalValence(), total_valence_vocab, allow_unknown=True) + \
           onehot(atom.GetFormalCharge(), formal_charge_vocab) + \
           [atom.GetIsAromatic()]



def atom_property_prediction(atom):
    """Property prediction atom feature.

    Features:
        GetSymbol(): one-hot embedding for the atomic symbol
        
        GetDegree(): one-hot embedding for the degree of the atom in the molecule
        
        GetTotalNumHs(): one-hot embedding for the total number of Hs (explicit and implicit) on the atom 
        
        GetTotalValence(): one-hot embedding for the total valence (explicit + implicit) of the atom
        
        GetFormalCharge(): one-hot embedding for the number of formal charges in the molecule
        
        GetIsAromatic(): whether the atom is aromatic
    """
    return onehot(atom.GetSymbol(), atom_vocab, allow_unknown=True) + \
           onehot(atom.GetDegree(), degree_vocab, allow_unknown=True) + \
           onehot(atom.GetTotalNumHs(), num_hs_vocab, allow_unknown=True) + \
           onehot(atom.GetTotalValence(), total_valence_vocab, allow_unknown=True) + \
           onehot(atom.GetFormalCharge(), formal_charge_vocab, allow_unknown=True) + \
           [atom.GetIsAromatic()]



def atom_position(atom):
    """
    Atom position in the molecular conformation.
    Return 3D position if available, otherwise 2D position is returned.

    Note it takes much time to compute the conformation for large molecules.
    """
    mol = atom.GetOwningMol()
    if mol.GetNumConformers() == 0:
        mol.Compute2DCoords()
    conformer = mol.GetConformer()
    pos = conformer.GetAtomPosition(atom.GetIdx())
    return [pos.x, pos.y, pos.z]



def atom_pretrain(atom):
    """Atom feature for pretraining.

    Features:
        GetSymbol(): one-hot embedding for the atomic symbol
        
        GetChiralTag(): one-hot embedding for atomic chiral tag
    """
    return onehot(atom.GetSymbol(), atom_vocab, allow_unknown=True) + \
           onehot(atom.GetChiralTag(), chiral_tag_vocab)



def atom_residue_symbol(atom):
    """Residue symbol as atom feature. Only support atoms in a protein.

    Features:
        GetSymbol(): one-hot embedding for the atomic symbol
        GetResidueName(): one-hot embedding for the residue symbol
    """
    residue = atom.GetPDBResidueInfo()
    return onehot(atom.GetSymbol(), atom_vocab, allow_unknown=True) + \
           onehot(residue.GetResidueName() if residue else -1, residue_vocab, allow_unknown=True)


def bond_default(bond):
    """Default bond feature.

    Features:
        GetBondType(): one-hot embedding for the type of the bond
        
        GetBondDir(): one-hot embedding for the direction of the bond
        
        GetStereo(): one-hot embedding for the stereo configuration of the bond
        
        GetIsConjugated(): whether the bond is considered to be conjugated
    """
    return onehot(bond.GetBondType(), bond_type_vocab, allow_unknown=True) + \
           onehot(bond.GetBondDir(), bond_dir_vocab) + \
           onehot(bond.GetStereo(), bond_stereo_vocab, allow_unknown=True) + \
           [int(bond.GetIsConjugated())]



def bond_length(bond):
    """
    Bond length in the molecular conformation.

    Note it takes much time to compute the conformation for large molecules.
    """
    mol = bond.GetOwningMol()
    if mol.GetNumConformers() == 0:
        mol.Compute2DCoords()
    conformer = mol.GetConformer()
    h = conformer.GetAtomPosition(bond.GetBeginAtomIdx())
    t = conformer.GetAtomPosition(bond.GetEndAtomIdx())
    return [h.Distance(t)]



def bond_property_prediction(bond):
    """Property prediction bond feature.

    Features:
        GetBondType(): one-hot embedding for the type of the bond
        
        GetIsConjugated(): whether the bond is considered to be conjugated
        
        IsInRing(): whether the bond is in a ring
    """
    return onehot(bond.GetBondType(), bond_type_vocab) + \
           [int(bond.GetIsConjugated()), bond.IsInRing()]



def bond_pretrain(bond):
    """Bond feature for pretraining.

    Features:
        GetBondType(): one-hot embedding for the type of the bond
        
        GetBondDir(): one-hot embedding for the direction of the bond
    """
    return onehot(bond.GetBondType(), bond_type_vocab) + \
           onehot(bond.GetBondDir(), bond_dir_vocab)



def residue_symbol(residue):
    """Symbol residue feature.

    Features:
        GetResidueName(): one-hot embedding for the residue symbol
    """
    return onehot(residue.GetResidueName(), residue_vocab, allow_unknown=True)



def residue_default(residue):
    """Default residue feature.

    Features:
        GetResidueName(): one-hot embedding for the residue symbol
    """
    return residue_symbol(residue)



def ExtendedConnectivityFingerprint(mol, radius=2, length=1024):
    """Extended Connectivity Fingerprint molecule feature.

    Features:
        GetMorganFingerprintAsBitVect(): a Morgan fingerprint for a molecule as a bit vector
    """
    ecfp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, length)
    return list(ecfp)




def molecule_default(mol):
    """Default molecule feature."""
    return ExtendedConnectivityFingerprint(mol)


ECFP = ExtendedConnectivityFingerprint

def get_full_connected_edge(frag):
    frag = np.asarray(list(frag))
    return torch.from_numpy(np.repeat(frag, len(frag)-1)), \
        torch.from_numpy(np.concatenate([np.delete(frag, i) for i in range(frag.shape[0])], axis=0))

def remove_repeat_edges(new_edge_index, refer_edge_index, N_atoms):
    new = to_dense_adj(new_edge_index, max_num_nodes=N_atoms)
    ref = to_dense_adj(refer_edge_index, max_num_nodes=N_atoms)
    delta_ = new - ref
    delta_[delta_<1] = 0
    unique, _ = dense_to_sparse(delta_)
    return unique

def get_ligand_feature(mol):
    xyz = mol.GetConformer().GetPositions()
    # covalent
    node_feature = []
    edge_index = []
    edge_feature = []
    for idx, atom in enumerate(mol.GetAtoms()):
        # node
        node_feature.append(atom_default(atom))
        # edge
        for bond in atom.GetBonds():
            edge_feature.append(bond_default(bond))
            for bond_idx in (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()):
                # if bond_idx != idx and bond_idx > idx:  # 单向
                if bond_idx != idx:  # 双向
                    edge_index.append([idx, bond_idx])
    node_feature = torch.from_numpy(np.asanyarray(node_feature)).float()  # nodes_chemical_features
    edge_index = torch.from_numpy(np.asanyarray(edge_index).T).long()
    edge_feature = torch.from_numpy(np.asanyarray(edge_feature)).float()
    # 0:4 bond type 5:11 bond direction 12:18 bond stero 19 bond conjunction
    l_full_edge_s = edge_feature[:, :5]
    xyz = torch.from_numpy(xyz)
    l_full_edge_s = torch.cat([l_full_edge_s, torch.pairwise_distance(xyz[edge_index[0]], xyz[edge_index[1]]).unsqueeze(dim=-1)], dim=-1)
    # ring
    adj = to_dense_adj(edge_index).squeeze(0)
    adj_ring = get_ring_adj_matrix(mol, adj)
    adj_ring = adj_ring - adj
    if adj_ring.any():
        (u_ring, v_ring), _ = dense_to_sparse(adj_ring)
        edge_index_new = torch.stack([u_ring, v_ring], dim=0)
        edge_index = torch.cat([edge_index, edge_index_new], dim=1)
        edge_feature_new = torch.zeros((edge_index_new.size(1), 20))
        edge_feature_new[:, [4, 5, 18]] = 1
        edge_feature = torch.cat([edge_feature, edge_feature_new], dim=0)
        l_full_edge_s_new = torch.cat([edge_feature_new[:, :5], torch.pairwise_distance(xyz[edge_index_new[0]], xyz[edge_index_new[1]]).unsqueeze(dim=-1)], dim=-1)
        l_full_edge_s = torch.cat([l_full_edge_s, l_full_edge_s_new], dim=0)
    # interaction
    adj_interaction = get_higher_order_adj_matrix(adj, order=3)
    adj_interaction = adj_interaction - adj
    (u_interaction, v_interaction), _ = dense_to_sparse(adj_interaction)
    edge_index_new = torch.stack([u_interaction, v_interaction], dim=0)
    edge_index = torch.cat([edge_index, edge_index_new], dim=1)
    edge_feature_new = torch.zeros((edge_index_new.size(1), 20))
    edge_feature_new[:, [4, 5, 18]] = 1
    edge_feature = torch.cat([edge_feature, edge_feature_new], dim=0)
    interaction_edge_mask = torch.ones((edge_feature.size(0),))
    interaction_edge_mask[-edge_feature_new.size(0):] = 0
    l_full_edge_s_new = torch.cat([edge_feature_new[:, :5], -torch.ones(edge_feature_new.size(0), 1)], dim=-1)
    l_full_edge_s = torch.cat([l_full_edge_s, l_full_edge_s_new], dim=0)
    x = (xyz, node_feature, edge_index, edge_feature, l_full_edge_s, interaction_edge_mask.bool()) 
    return x 

def get_ligand_feature_v1(mol, use_chirality=True):
    xyz = mol.GetConformer().GetPositions()
    # covalent
    N_atoms = mol.GetNumAtoms()
    node_feature = []
    edge_index = []
    edge_feature = []
    G = nx.Graph()
    for idx, atom in enumerate(mol.GetAtoms()):
        # node
        node_feature.append(atom_default(atom))
        # edge
        for bond in atom.GetBonds():
            edge_feature.append(bond_default(bond))
            for bond_idx in (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()):
                # if bond_idx != idx and bond_idx > idx:  # 单向
                if bond_idx != idx:  # 双向
                    edge_index.append([idx, bond_idx])
                    G.add_edge(idx, bond_idx)
    node_feature = torch.from_numpy(np.asanyarray(node_feature)).float()  # nodes_chemical_features
    if use_chirality:
        try:
            chiralcenters = Chem.FindMolChiralCenters(mol,force=True,includeUnassigned=True, useLegacyImplementation=False)
        except:
            chiralcenters = []
        chiral_arr = torch.zeros([N_atoms,3]) 
        for (i, rs) in chiralcenters:
            if rs == 'R':
                chiral_arr[i, 0] =1 
            elif rs == 'S':
                chiral_arr[i, 1] =1 
            else:
                chiral_arr[i, 2] =1 
        node_feature = torch.cat([node_feature,chiral_arr.float()],dim=1)
    edge_index = torch.from_numpy(np.asanyarray(edge_index).T).long()
    edge_feature = torch.from_numpy(np.asanyarray(edge_feature)).float()
    cov_edge_num = edge_index.size(1)
    # 0:4 bond type 5:11 bond direction 12:18 bond stero 19 bond conjunction
    l_full_edge_s = edge_feature[:, :5]
    xyz = torch.from_numpy(xyz)
    l_full_edge_s = torch.cat([l_full_edge_s, torch.pairwise_distance(xyz[edge_index[0]], xyz[edge_index[1]]).unsqueeze(dim=-1)], dim=-1)
    # get fragments based on rotation bonds
    frags = []
    rotate_bonds = []
    for e in G.edges():
        G2 = copy.deepcopy(G)
        G2.remove_edge(*e)
        if nx.is_connected(G2): continue
        # print(f'{sorted(nx.connected_components(G2), key=len)[0]}|{sorted(nx.connected_components(G2), key=len)[1]}')
        l = (sorted(nx.connected_components(G2), key=len)[0])
        if len(l) < 2: continue
        else:
            # rotate_bonds.append(e)
            # rotate_bonds.append((e[1], e[0]))
            frags.append(l)
    if len(frags) != 0:
        frags = sorted(frags, key=len)
        for i in range(len(frags)):
            for j in range(i+1, len(frags)):
                frags[j] -= frags[i]
        frags = [i for i in frags if len(i) > 1]
        frag_edge_index = torch.cat([torch.stack(get_full_connected_edge(i), dim=0) for i in frags], dim=1).long()
        edge_index_new = remove_repeat_edges(new_edge_index=frag_edge_index, refer_edge_index=edge_index, N_atoms=N_atoms)
        edge_index = torch.cat([edge_index, edge_index_new], dim=1)
        edge_feature_new = torch.zeros((edge_index_new.size(1), 20))
        edge_feature_new[:, [4, 5, 18]] = 1
        edge_feature = torch.cat([edge_feature, edge_feature_new], dim=0)
        l_full_edge_s_new = torch.cat([edge_feature_new[:, :5], torch.pairwise_distance(xyz[edge_index_new[0]], xyz[edge_index_new[1]]).unsqueeze(dim=-1)], dim=-1)
        l_full_edge_s = torch.cat([l_full_edge_s, l_full_edge_s_new], dim=0)
    # interaction
    adj_interaction = torch.ones((N_atoms, N_atoms)) - torch.eye(N_atoms, N_atoms)
    interaction_edge_index, _ = dense_to_sparse(adj_interaction)
    edge_index_new = remove_repeat_edges(new_edge_index=interaction_edge_index, refer_edge_index=edge_index, N_atoms=N_atoms)
    edge_index = torch.cat([edge_index, edge_index_new], dim=1)
    edge_feature_new = torch.zeros((edge_index_new.size(1), 20))
    edge_feature_new[:, [4, 5, 18]] = 1
    edge_feature = torch.cat([edge_feature, edge_feature_new], dim=0)
    interaction_edge_mask = torch.ones((edge_feature.size(0),))
    interaction_edge_mask[-edge_feature_new.size(0):] = 0
    # scale the distance
    l_full_edge_s[:, -1] *= 0.1
    l_full_edge_s_new = torch.cat([edge_feature_new[:, :5], -torch.ones(edge_feature_new.size(0), 1)], dim=-1)
    l_full_edge_s = torch.cat([l_full_edge_s, l_full_edge_s_new], dim=0)
    # cov edge mask
    cov_edge_mask = torch.zeros(edge_feature.size(0),)
    cov_edge_mask[:cov_edge_num] = 1
    x = (xyz, node_feature, edge_index, edge_feature, l_full_edge_s, interaction_edge_mask.bool(), cov_edge_mask.bool()) 
    return x 
