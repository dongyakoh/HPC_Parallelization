# -*- coding: utf-8 -*-
"""
Created on Sat Sep  8 00:12:12 2018
Value Function Iteration

@author: Don Koh
"""

import matplotlib.pyplot as plt
plt.rcdefaults()
plt.style.use('seaborn-bright')
import numpy as np
import json
from scipy.optimize import bisect



oop_flag    = 0     # 0: oop, 1: pop
q_flag      = 2     # 0: med16core, 1: onenode16core, 2: mem768GB32core

if oop_flag is 0:
    oop_var = "_oop"
else:
    oop_var = "_pop"
    
if q_flag is 0:
    q_var = "_med16core"
    max_core    = 16
elif q_flag is 1:
    q_var = "_tiny16core"
    max_core    = 16
elif q_flag is 2:
    q_var = "_mem768GB32core"
    max_core    = 32
elif q_flag is 3:
    q_var = "_uma"
    max_core    = 8
    
    
file_name = "runtime_vfi_by_cores" + oop_var + q_var + ".txt"

list_cores      = np.arange(1,max_core+1,1)
num_list_cores  = len(list_cores)


nrun        = 21
#nrun        = 2

# Read data from file
with open(file_name) as json_data:
    d = json.load(json_data)


x_cores     = np.zeros((num_list_cores,nrun-1))
tp_n        = np.zeros((num_list_cores,nrun-1))
index = 0
for ik in range(len(d)):
    if d[ik]['age']==0 and d[ik]['run']>0 and d[ik]['core']>0:
        x_cores[index,d[ik]['run']-1] = d[ik]['core']
        tp_n[index,d[ik]['run']-1] = d[ik]['runtime']
        if d[ik]['run']==nrun-1:
            index += 1

# Total runtime of parallelizable code
tp_n_max    = np.amax(tp_n,axis=1)
tp_n_min    = np.amin(tp_n,axis=1)
tp_n_med    = np.median(tp_n,axis=1)

p_n         = np.zeros(tp_n.shape)
p_n_max     = np.zeros(tp_n_max.shape)
p_n_min     = np.zeros(tp_n_min.shape)
p_n_med     = np.zeros(tp_n_med.shape)

sp_n         = np.zeros(tp_n.shape)
sp_n_max     = np.zeros(tp_n_max.shape)
sp_n_min     = np.zeros(tp_n_min.shape)
sp_n_med     = np.zeros(tp_n_med.shape)

ef_n         = np.zeros(tp_n.shape)
ef_n_max     = np.zeros(tp_n_max.shape)
ef_n_min     = np.zeros(tp_n_min.shape)
ef_n_med     = np.zeros(tp_n_med.shape)


# Parallelization overhead
for ncores in list_cores:
    p_n[ncores-1,:]       = tp_n[ncores-1,:] - tp_n[0,:]/ncores
    p_n_max[ncores-1]     = tp_n_max[ncores-1] - tp_n_max[0]/ncores
    p_n_min[ncores-1]     = tp_n_min[ncores-1] - tp_n_min[0]/ncores
    p_n_med[ncores-1]     = tp_n_med[ncores-1] - tp_n_med[0]/ncores

    sp_n[ncores-1,:]       = tp_n[0,:]/tp_n[ncores-1,:]
    sp_n_max[ncores-1]     = tp_n_max[0]/tp_n_max[ncores-1]
    sp_n_min[ncores-1]     = tp_n_min[0]/tp_n_min[ncores-1]
    sp_n_med[ncores-1]     = tp_n_med[0]/tp_n_med[ncores-1]

    ef_n[ncores-1,:]       = sp_n[ncores-1,:]/ncores
    ef_n_max[ncores-1]     = sp_n_max[ncores-1]/ncores
    ef_n_min[ncores-1]     = sp_n_min[ncores-1]/ncores
    ef_n_med[ncores-1]     = sp_n_med[ncores-1]/ncores

        

tp_n_med_norm = 100*tp_n_med/tp_n_med[0]
p_n_med_norm = 100*p_n_med/tp_n_med[0]


# Plot
####################################

fs  = 12
wid = 0.8

fig, ax0 = plt.subplots(figsize=(7, 5))
p1 = plt.plot(list_cores[1:], p_n_med_norm[1:], '-o')
plt.xlabel('# of Cores', color = 'black', fontsize=fs)
plt.ylabel('% of Total Serial Runtime', color = 'black', fontsize=fs)
plt.xticks(list_cores[1:],color = 'black',fontsize = fs, rotation=90)
plt.yticks(np.arange(0,2.51,0.25),color = 'black',fontsize = fs)
plt.grid()
ax0.xaxis.grid(linestyle=':')
ax0.yaxis.grid(linestyle=':')
plt.tight_layout()
fig_name = "po_by_cores" + oop_var + "_1_" + str(max_core) + ".png"
plt.savefig(fig_name, dpi = 300, bbox_inches = "tight")
plt.show()


fig, ax1 = plt.subplots(figsize=(7, 5))
p1 = plt.plot(list_cores[1:], sp_n_med[1:], '-o')
p2 = plt.plot(list_cores[1:], list_cores[1:], '--', color='gray')
plt.xlabel('# of Cores', color = 'black', fontsize=fs)
plt.ylabel('Relative Performance', color = 'black', fontsize=fs)
plt.xticks(list_cores[1:],color = 'black',fontsize = fs, rotation=90)
plt.yticks(np.arange(2,33,2),color = 'black',fontsize = fs)
plt.grid()
ax1.xaxis.grid(linestyle=':')
ax1.yaxis.grid(linestyle=':')
plt.tight_layout()
plt.text(30, 31, "45$^{\circ}$",{'color': 'k', 'fontsize': 12, 'ha': 'center', 'va': 'center'})
#plt.legend(loc='lower right', fontsize = fs)
fig_name = "speedup_by_cores" + oop_var + "_1_" + str(max_core) + ".png"
plt.savefig(fig_name, dpi = 300, bbox_inches = "tight")
plt.show()

"""
fig, ax2 = plt.subplots(figsize=(7, 5))
p1 = plt.plot(list_cores[1:], ef_n_med[1:], '-o')
plt.xlabel('Number of Cores', color = 'black', fontsize=fs)
plt.ylabel('Efficiency', color = 'black', fontsize=fs)
plt.xticks(list_cores[1:],color = 'black',fontsize = fs, rotation=90)
plt.yticks(np.arange(0.5,1.1,0.1),color = 'black',fontsize = fs)
plt.grid()
ax2.xaxis.grid(linestyle=':')
ax2.yaxis.grid(linestyle=':')
plt.tight_layout()
fig_name = "efficiency_by_cores" + oop_var + "_1_" + str(max_core) + ".png"
plt.savefig(fig_name, dpi = 300, bbox_inches = "tight")
plt.show()
"""