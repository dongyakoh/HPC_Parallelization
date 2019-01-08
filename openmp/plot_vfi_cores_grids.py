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
q_flag      = 1    # 0: med16core, 1: onenode16core, 2: mem768GB32core

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
    z_tp_n_max[idx_core,:]  = np.amax(y_grids,axis=1)
    z_tp_n_min[idx_core,:]  = np.amin(y_grids,axis=1)
    z_tp_n_med[idx_core,:]  = np.median(y_grids,axis=1)
    
    z_p_n_max[idx_core,:]   = z_tp_n_max[idx_core,:] - z_tp_n_max[0,:]/num_core
    z_p_n_min[idx_core,:]   = z_tp_n_min[idx_core,:] - z_tp_n_min[0,:]/num_core
    z_p_n_med[idx_core,:]   = z_tp_n_med[idx_core,:] - z_tp_n_med[0,:]/num_core
    
    z_tp_n_norm[idx_core,:]     = z_tp_n_med[idx_core,:]/z_tp_n_med[0,:]
    z_p_n_norm[idx_core,:]      = 100*z_p_n_med[idx_core,:]/z_tp_n_med[0,:]
    
"""
#################################################
# Serial Runtime
file_name1 = "runtime_vfi_by_cores_grids" + oop_var + "_1_" + str(max_core) + ".txt"
nrun1        = 2

# Read data from file
with open(file_name1) as json_data:
    d1 = json.load(json_data)

z_tp_1        = np.zeros(num_list_grids)

grid_index = 0
for ik in range(len(d1)):
    if d1[ik]['core']==1 and d1[ik]['age']==0 and d1[ik]['run']==1:
        z_tp_1[grid_index] = d1[ik]['runtime']
        grid_index += 1


for idx_core, num_core in enumerate(list_cores):
    z_p_n_max[idx_core,:]       = z_tp_n_max[idx_core,:] - z_tp_1/num_core
    z_p_n_min[idx_core,:]       = z_tp_n_min[idx_core,:] - z_tp_1/num_core
    z_p_n_med[idx_core,:]       = z_tp_n_med[idx_core,:] - z_tp_1/num_core
    
    z_tp_n_norm[idx_core,:]     = z_tp_n_med[idx_core,:]/z_tp_1
    z_p_n_norm[idx_core,:]      = 100*z_p_n_med[idx_core,:]/z_tp_1
"""


# Plot
####################################
X_grids, Y_cores = np.meshgrid(list_grids,list_cores[1:])

fs  = 10    # Font Size
fig     = plt.figure(figsize=(8, 5))
ax      = fig.gca(projection='3d')
surf = ax.plot_surface(X_grids, Y_cores, z_p_n_norm[1:],cmap=cm.coolwarm,linewidth=0)
ax.set_xlabel('Grid Size', color = 'black', fontsize = fs)
ax.set_ylabel('Number of Cores', color = 'black', fontsize = fs)
#ax.set_zlabel('% of Total Serial Runtime', color = 'black', fontsize = fs)
ax.set_xticks(list_grids)
ax.set_yticks(np.arange(2,33,1))
majors = np.arange(2,33,2)
ax.yaxis.set_major_locator(ticker.FixedLocator(majors))

fig.colorbar(surf, shrink=0.5, aspect=5)

ax.view_init(50, 200)
plt.savefig(fig_name,dpi=300)
plt.show()
