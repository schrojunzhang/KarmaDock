# KarmaDock: a deep learning paradigm for ultra-large library docking with fast speed and high accuracy

![](https://github.com/schrojunzhang/KarmaDock/blob/main/result1.gif)

## Contents

- [Overview](#overview)
- [Software Requirements](#software-requirements)
- [Installation Guide](#installation-guide)
- [Demo](#demo--reproduction-ligand-docking-on-pdbbind-core-set)

## Overview 

KarmaDock is a deep learning framework that enables ligand docking with fast speed and high accuracy. The framework consists of four main steps: creating Python environments, preprocessing PDBBind data, generating graphs based on protein-ligand complexes, and ligand docking.

## Software Requirements

### OS Requirements

The package development version is tested on *Linux: Ubuntu 18.04* operating systems.

### Python Dependencies

Dependencies for KarmaDock:

```
pytorch
pyg
rdkit
mdanalysis
prody 
```

## Installation Guide

### download this repo

```
git clone https://github.com/schrojunzhang/KarmaDock.git
```

### install karmadock_env

you can install the env via yaml file

```
cd KarmaDock
conda env create -f karmadock_env.yaml
```

or you can download the [conda-packed file](https://zenodo.org/record/7788732/files/karmadock_env.tar.gz?download=1), and then unzip it in `${anaconda install dir}/anaconda3/envs`. `${anaconda install dir}` represents the dir where the anaconda is installed. For me, ${anaconda install dir}=/root . 

```
mkdir ${anaconda install dir}/anaconda3/envs/karmadock 
tar -xzvf karmadock.tar.gz -C ${anaconda install dir}/anaconda3/envs/karmadock
conda activate karmadock
```

## Demo & Reproduction: ligand docking on PDBBind core set

Assume that the project is at `/root` and therefore the project path is /root/KarmaDock.

### 1. Download PDBBind dataset

You can download the PDBBind 2020 core set without preprocessing from the [PDBBind website](http://pdbbind.org.cn/index.php)
OR you can download [the version](https://zenodo.org/record/7788083/files/pdbbind2020_core_set.zip?download=1) where protein files were prepared by Schrodinger. 
```
cd /root/KarmaDock
weget https://zenodo.org/record/7788083/files/pdbbind2020_core_set.zip?download=1
unzip -q pdbbind2020_core_set.zip?download=1
```

### 2. Preprocess PDBBind data

The purpose of this step is to identify residues that are within a 12Å radius of any ligand atom and use them as the pocket of the protein. The pocket file (xxx_pocket_ligH12A.pdb) will also be saved on the `complex_file_dir`.

```
cd /root/KarmaDock/utils 
python -u pre_processing.py --complex_file_dir ~/your/PDBBindDataset/path
```
e.g.,
```
cd /root/KarmaDock/utils 
python -u pre_processing.py --complex_file_dir /root/KarmaDock/pdbbind2020_core_set
```

### 3. Generate graphs based on protein-ligand complexes

This step will generate graphs for protein-ligand complexes and save them (*.dgl) to `graph_file_dir`.

```
cd /root/KarmaDock/utils 
python -u generate_graph.py 
--complex_file_dir ~/your/PDBBindDataset/path 
--graph_file_dir ~/the/directory/for/saving/graph 
--csv_file ~/path/of/csvfile/with/dataset/split
```
e.g.,
```
mkdir /root/KarmaDock/test_graph
cd /root/KarmaDock/utils 
python -u generate_graph.py --complex_file_dir /root/KarmaDock/pdbbind2020_core_set --graph_file_dir /root/KarmaDock/test_graph --csv_file /root/KarmaDock/pdbbind2020.csv
```

### 4. ligand docking

This step will perform ligand docking (predict binding poses and binding strengthes) based on the graphs. (finished in about 0.5 min)

```
cd /root/KarmaDock/utils 
python -u ligand_docking.py 
--graph_file_dir ~/the/directory/for/saving/graph 
--csv_file ~/path/of/csvfile/with/dataset/split 
--model_file ~/path/of/trained/model/parameters 
--out_dir ~/path/for/recording/BindingPoses&DockingScores 
--docking Ture/False  whether generating binding poses
--scoring Ture/False  whether predict binding affinities
--correct Ture/False  whether correct the predicted binding poses
--batch_size 64 
--random_seed 2023 
```
e.g.,
```
mkdir /root/KarmaDock/test_result
cd /root/KarmaDock/utils 
python -u ligand_docking.py --graph_file_dir /root/KarmaDock/test_graph --csv_file /root/KarmaDock/pdbbind2020.csv --model_file /root/KarmaDock/trained_models/karmadock_docking.pkl --out_dir /root/KarmaDock/test_result --docking True --scoring True --correct True --batch_size 64 --random_seed 2023
```
