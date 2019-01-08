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
from matplotlib import cm
import matplotlib.ticker as ticker



oop_flag    = 0     # 0: oop, 1: pop
q_flag      = 1    # 0: med16core, 1: mem768GB32core

if oop_flag is 0:
    oop_var = "_oop"
else:
    oop_var = "_pop"
    
if q_flag is 0:
    q_var = "_med16core"
    max_core    = 16
    list_cores      = [1,2,4,8,16]
elif q_flag is 1:
    q_var = "_mem768GB32core"
    max_core    = 32
    list_cores      = [1,2,4,8,16,32]
    
#list_cores      = np.arange(2,max_core+1,1)
list_grids      = np.arange(500,1501,100)
num_list_cores  = len(list_cores)
num_list_grids  = len(list_grids)
nrun0            = 3

x_grids         = np.zeros(num_list_grids)
y_grids         = np.zeros((num_list_grids,nrun0))
z_tp_n_max      = np.zeros((num_list_cores,num_list_grids))
z_tp_n_min      = np.zeros((num_list_cores,num_list_grids))
z_tp_n_med      = np.zeros((num_list_cores,num_list_grids))
z_tp_n_norm     = np.zeros((num_list_cores,num_list_grids))
z_p_n_max       = np.zeros((num_list_cores,num_list_grids))
z_p_n_min       = np.zeros((num_list_cores,num_list_grids))
z_p_n_med       = np.zeros((num_list_cores,num_list_grids))
z_p_n_norm      = np.zeros((num_list_cores,num_list_grids))


fig_name = "vfi_by_cores_grids" + oop_var + "_1_" + str(max_core) + ".png"



for idx_core, num_core in enumerate(list_cores):

    # Read data from file
    file_name0 = "runtime_vfi_by_cores_grids" + oop_var + q_var + "_" + str(num_core) + ".txt"
    with open(file_name0) as json_data:
        d0 = json.load(json_data)

    # Total runtime of parallelizable code
    grid_index = 0
    for ik in range(len(d0)):
        if d0[ik]['age']==0 and d0[ik]['run']<nrun0:
            x_grids[grid_index]                 = d0[ik]['grid']
            y_grids[grid_index,d0[ik]['run']]   = d0[ik]['runtime']
            if d0[ik]['run']==nrun0-1:
                grid_index += 1
                
    # Total runtime of parallelizable code
    y_max    = np.amax(y_grids,axis=1)
    y_min    = np.amin(y_grids,axis=1)
    y_med    = np.median(y_grids,axis=1)

    z_tp_n_max[idx_core,:]  = y_max
    z_tp_n_min[idx_core,:]  = y_min
    z_tp_n_med[idx_core,:]  = y_med

    z_p_n_max[idx_core,:]   = z_tp_n_max[idx_core,:] - z_tp_n_max[0,:]/num_core
    z_p_n_min[idx_core,:]   = z_tp_n_min[idx_core,:] - z_tp_n_min[0,:]/num_core
    z_p_n_med[idx_core,:]   = z_tp_n_med[idx_core,:] - z_tp_n_med[0,:]/num_core
    
    z_tp_n_norm[idx_core,:]     = z_tp_n_med[idx_core,:]/z_tp_n_med[0,:]
    z_p_n_norm[idx_core,:]      = 100*z_p_n_med[idx_core,:]/z_tp_n_med[0,:]


# Plot
####################################

fs  = 10    # Font Size

fig, ax = plt.subplots(figsize=(7, 5))

ax.plot(list_grids, z_p_n_norm[1,:], '-o',label="2 cores")
ax.plot(list_grids, z_p_n_norm[2,:], '-+',label="4 cores")
ax.plot(list_grids, z_p_n_norm[3,:], '-d',label="8 cores")
ax.plot(list_grids, z_p_n_norm[4,:], '-^',label="16 cores")
if q_flag is 1:
    ax.plot(list_grids, z_p_n_norm[5,:], '-s',label="32 cores")


# Axes labels
plt.xlabel('Grid Size', color = 'black', fontsize=fs)
plt.ylabel('% of Total Serial Runtime', color = 'black', fontsize=fs)

# Background grid
ax.xaxis.grid(which='major',linestyle=':')
ax.xaxis.grid(which='minor',linestyle=':')
ax.yaxis.grid(linestyle=':')

# Legend
plt.legend(loc='upper right', fontsize = fs)
plt.savefig(fig_name, dpi = 300)
plt.show()
