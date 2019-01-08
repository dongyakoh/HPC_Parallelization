# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 12:14:31 2018

@author: dkoh
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.rcdefaults()
plt.style.use('seaborn-bright')
import numpy as np
import json



oop_flag    = 1     # 0: oop, 1: pop
q_flag      = 1    # 0: med16core, 1: mem768GB32core

if oop_flag is 0:
    oop_var = "_oop"
else:
    oop_var = "_pop"
    
if q_flag is 0:
    q_var = "_med16core"
    max_core    = 16
elif q_flag is 1:
    q_var = "_mem768GB32core"
    max_core    = 32

list_cores      = np.arange(1,max_core+1,1)
list_grids      = np.arange(500,1501,100)
num_list_cores  = len(list_cores)
num_list_grids  = len(list_grids)

# Rewrite the file


for idx_core, num_core in enumerate(list_cores):

    file_name0 = "runtime_vfi_by_cores_grids" + oop_var + "_1_" + str(max_core) + "_" + str(num_core) + ".txt"
    
    # Read data from file
    with open(file_name0) as json_data:
        d0 = json.load(json_data)
    
    # Iterate on the number of cores
    nrun        = 3
    
    data = []
    
    # Total runtime of parallelizable code
    for ik in range(len(d0)):
        if d0[ik]['age']==0 and d0[ik]['run']<nrun:
        
            data.append({
                'core': d0[ik]['core'],
                'grid': d0[ik]['grid'],
                'run': d0[ik]['run'],
                'age': d0[ik]['age'],
                'runtime': d0[ik]['runtime']
            })

    file_out_name = "runtime_vfi_by_cores_grids" + oop_var + q_var + "_" + str(num_core) + ".txt"
    with open(file_out_name,"w+") as outfile:
        json.dump(data, outfile)



