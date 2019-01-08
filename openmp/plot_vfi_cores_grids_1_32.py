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

oop_flag    = 0     # 0: oop, 1: pop
q_flag      = 1     # 0: med16core, 1: mem768GB32core

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
    
    
file_name = "runtime_vfi_by_cores_grids" + oop_var + q_var + ".txt"
fig_name = "vfi_by_cores" + oop_var + "_1_" + str(max_core) + ".png"

list_cores      = np.arange(1,max_core+1,1)
list_grids      = np.arange(500,1501,100)
num_list_cores  = len(list_cores)
num_list_grids  = len(list_grids)

# Read data from file
with open(file_name) as json_data:
    d = json.load(json_data)


x_grids         = list_grids
y_cores         = list_cores
z_tp_n          = np.zeros(num_list_cores)
z_p_n           = np.zeros(num_list_cores)
z_tp_n_norm     = np.zeros(num_list_cores)
z_p_n_norm      = np.zeros(num_list_cores)


# Total runtime of parallelizable code
core_index = 0
grid_index = 0
for ik in range(len(d)):
    if d[ik]['age']==0 and d[ik]['run']==1 and d[ik]['core']>0 and d[ik]['core']<=list_cores[num_list_cores-1] and d[ik]['grid']==1000:
        z_tp_n[core_index]   = d[ik]['runtime']
        core_index += 1

# Parallelization overhead
for ncores in list_cores:
    z_p_n[ncores-1]       = z_tp_n[ncores-1] - z_tp_n[0]/ncores
    z_p_n_norm[ncores-1] = z_p_n[ncores-1]/z_tp_n[0]
    z_tp_n_norm[ncores-1] = z_tp_n[ncores-1]/z_tp_n[0]



# Plot
####################################
fs  = 14
wid = 0.8

fig, ax = plt.subplots(figsize=(7, 5))

p1 = plt.fill_between(list_cores, 0, z_tp_n_norm, facecolor='green',label="Parallelized Tasks")
p2 = plt.fill_between(list_cores, 0, z_p_n_norm, facecolor='darkorange',label="Parallelization Overhead")

# Axes labels
plt.xlabel('Number of Cores', color = 'black', fontsize=fs)
plt.ylabel('% of Total Serial Runtime', color = 'black', fontsize=fs)

# Axes ticks
plt.xticks(list_cores, color = 'black',fontsize = fs, rotation=90)
plt.yticks(np.arange(0,1.1,0.1), color = 'black',fontsize = fs)

# Background grid
plt.grid()
ax.xaxis.grid(linestyle=':')
ax.yaxis.grid(linestyle=':')

# Legend
plt.legend(loc='upper right', fontsize = fs)
plt.savefig(fig_name,dpi=300)
plt.show()