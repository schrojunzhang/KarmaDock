# KarmaDock: a deep learning paradigm for ultra-large library docking with fast speed and high accuracy

![](https://github.com/schrojunzhang/KarmaDock/blob/main/result1.gif)

## Contents

- [Overview](#overview)
- [Software Requirements](#software-requirements)
- [Installation Guide](#installation-guide)
- [Demo: Ligand docking](#demo1--ligand-docking-on-pdbbind-core-set)
- [Demo: Virtual screening](#demo2--virtual-screening-on-dekois-20)

## Overview

Ligand docking is one of the core technologies in structure-based virtual screening for drug discovery. However, conventional docking tools and existing deep learning tools may suffer from limited performance in terms of speed, pose quality and binding affinity accuracy. Here we propose KarmaDock, a deep learning approach for ligand docking that integrates the functions of docking acceleration, binding pose generation and correction, and binding strength estimation. The three-stage model consists of the following components: (1) encoders for the protein and ligand to learn the representations of intramolecular interactions; (2) E(n) equivariant graph neural networks with self-attention to update the ligand pose based on both protein–ligand and intramolecular interactions, followed by post-processing to ensure chemically plausible structures; (3) a mixture density network for scoring the binding strength. KarmaDock was validated on four benchmark datasets and tested in a real-world virtual screening project that successfully identified experiment-validated active inhibitors of leukocyte tyrosine kinase (LTK).

If you find it useful, please cite:

**Efficient and accurate large library ligand docking with KarmaDock**
Zhang, Xujun#; Zhang, Odin#; Shen, Chao; Qu, Wanglin; Chen, Shicheng; Cao, Hanqun; Kang, Yu; Wang, Zhe; Wang, Ercheng; Zhang, Jintu; Deng, Yafeng; Liu, Furui; Wang, Tianyue; Du, Hongyan; Wang, Langcheng; Pan, Peichen*; Chen, Guangyong*; Hsieh, Chang-Yu*; Hou, Tingjun*.
*Published in:* Nature Computational Science, 2023, Vol. 3, No. 9, pp. 789-804.
*DOI:* [10.1038/s43588-023-00511-5](https://www.nature.com/articles/s43588-023-00511-5)

## Software Requirements

### OS Requirements

The package development version is tested on *Linux: Ubuntu 18.04* operating systems.

### Python Dependencies

Dependencies for KarmaDock:

```
pytorch
pyg
rdkit=2022.09.1 (important!!!)
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

## Demo1 & ligand docking on PDBBind core set

Assume that the project is at `/root` and therefore the project path is /root/KarmaDock.

### 1. Download PDBBind dataset

You can download the PDBBind 2020 core set without preprocessing from the [PDBBind website](http://pdbbind.org.cn/index.php)
OR you can download [the version](https://zenodo.org/record/7788083/files/pdbbind2020_core_set.zip?download=1) where protein files were prepared by Schrodinger.

```
cd /root/KarmaDock
wget https://zenodo.org/record/7788083/files/pdbbind2020_core_set.zip?download=1
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
```

e.g.,

```
cd /root/KarmaDock/utils 
python -u generate_graph.py --complex_file_dir /root/KarmaDock/pdbbind2020_core_set --graph_file_dir /root/KarmaDock/pdbbind_graph 
```

### 4. ligand docking

This step will perform ligand docking (predict binding poses and binding strengthes) based on the graphs. (finished in about 0.5 min)

```
cd /root/KarmaDock/utils 
python -u ligand_docking.py 
--graph_file_dir ~/the/directory/for/saving/graph 
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
cd /root/KarmaDock/utils 
python -u ligand_docking.py --graph_file_dir /root/KarmaDock/pdbbind_graph --model_file /root/KarmaDock/trained_models/karmadock_screening.pkl --out_dir /root/KarmaDock/pdbbind_result --docking True --scoring True --correct True --batch_size 64 --random_seed 2023
```

## Demo2 & virtual screening on DEKOIS 2.0

Assume that the project is at `/root` and therefore the project path is /root/KarmaDock.

### 1. Download DEKOIS dataset

You can download the DEKOIS 2.0 dataset without preprocessing from the [DEKOIS website](http://www.pharmchem.uni-tuebingen.de/dekois/)
OR you can download [the version](https://zenodo.org/record/8131256/files/DEKOIS2.zip?download=1) where protein files were prepared by Schrodinger, glide-docked poses were provided.
```
cd /root/KarmaDock
wget https://zenodo.org/record/8131256/files/DEKOIS2.zip?download=1
unzip -q DEKOIS2.zip?download=1
```
### 2. virtual screening

This step will perform virtual screening for a specific target PDK1 (predict binding poses and binding strengthes).

###### (1) CPU and GPU machines (faster):
You can run the following command on CPUs before performing virtual screening (generate graphs in advance)
```
cd /root/KarmaDock/utils
python -u virtual_screening_pipeline.py 
--mode generate_graph
--ligand_smi ~/the/directory/for/ligand/library/smi 
--protein_file ~/the/directory/for/target/protein/pdb 
--crystal_ligand_file ~/the/directory/for/crystal/ligand/mol2/for/binding/pocket 
--graph_dir ~/the/directory/for/saving/ligand/graphs 
--random_seed 2023 
```
e.g.,

```
cd /root/KarmaDock/utils 
python -u virtual_screening_pipeline.py --ligand_smi /root/KarmaDock/DEKOIS2/pdk1/active_decoys.smi --protein_file /root/KarmaDock/DEKOIS2/pdk1/protein/pdk1_protein.pdb --crystal_ligand_file /root/KarmaDock/DEKOIS2/pdk1/protein/pdk1_ligand.mol2 --graph_dir /root/KarmaDock/DEKOIS2/pdk1/karmadock_liggraph --random_seed 2023 
```
Then, you can run the following command on GPUs to perform virtual screening (predict binding poses and binding strengthes)
```
cd /root/KarmaDock/utils
python -u virtual_screening_pipeline.py 
--mode vs
--protein_file ~/the/directory/for/target/protein/pdb 
--crystal_ligand_file ~/the/directory/for/crystal/ligand/mol2/for/binding/pocket 
--graph_dir ~/the/directory/for/saving/ligand/graphs 
--out_dir ~/path/for/recording/BindingPoses&DockingScores 
--score_threshold 50 
--batch_size 64 
--random_seed 2023 
--out_uncoorected 
--out_corrected
```
e.g.,

```
cd /root/KarmaDock/utils 
python -u virtual_screening_pipeline.py --protein_file /root/KarmaDock/DEKOIS2/pdk1/protein/pdk1_protein.pdb --crystal_ligand_file /root/KarmaDock/DEKOIS2/pdk1/protein/pdk1_ligand.mol2 --graph_dir /root/KarmaDock/DEKOIS2/pdk1/karmadock_liggraph --out_dir /root/KarmaDock/DEKOIS2/pdk1/karmadocked --score_threshold 50 --batch_size 64 --random_seed 2023 --out_uncoorected --out_corrected
```
###### (2) GPU machines (slower but more convinent):
For pure GPU machines, you can run the following command to perform virtual screening (generate graphs on the fly)
```
cd /root/KarmaDock/utils 
python -u virtual_screening.py 
--ligand_smi ~/the/directory/for/ligand/library/smi 
--protein_file ~/the/directory/for/target/protein/pdb 
--crystal_ligand_file ~/the/directory/for/crystal/ligand/mol2/for/binding/pocket 
--out_dir ~/path/for/recording/BindingPoses&DockingScores 
--score_threshold 50
--batch_size 64 
--random_seed 2023 
--out_uncoorected
--out_corrected
```

e.g.,

```
cd /root/KarmaDock/utils 
python -u virtual_screening.py --ligand_smi /root/KarmaDock/DEKOIS2/pdk1/active_decoys.smi --protein_file /root/KarmaDock/DEKOIS2/pdk1/protein/pdk1_protein.pdb --crystal_ligand_file /root/KarmaDock/DEKOIS2/pdk1/protein/pdk1_ligand.mol2 --out_dir /root/KarmaDock/DEKOIS2/pdk1/karmadocked --score_threshold 50 --batch_size 64 --random_seed 2023 --out_uncoorected --out_corrected
```
