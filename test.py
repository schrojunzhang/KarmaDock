#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   test.py
@Time    :   2023/03/05 16:18:56
@Author  :   Xujun Zhang
@Version :   1.0
@Contact :   
@License :   
@Desc    :   None
'''
# here put the import lib
import copy
import pandas as pd
import shutil
import os

df = pd.read_csv('./pdbbind2020.csv')
df_core = df[df.group == 'core']
pdb_ids = df_core.pdb_id.values
totol_pdb_ids = os.listdir(r"C:\Users\xujunzhang\Desktop\pdbbind2020")
for pdb_id in totol_pdb_ids:
    if pdb_id not in pdb_ids:
        shutil.rmtree(rf'C:\Users\xujunzhang\Desktop\pdbbind2020\{pdb_id}')
    else:
        try:
            os.remove(fr'C:\Users\xujunzhang\Desktop\pdbbind2020\{pdb_id}\{pdb_id}_pocket_ligH12A.pdb')
        except:
            pass
