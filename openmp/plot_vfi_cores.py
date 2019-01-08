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



oop_flag    = 1     # 0: oop, 1: pop
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
fig_name = "vfi_by_cores" + oop_var + "_1_" + str(max_core) + ".png"

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
dp_n_f      = np.zeros(tp_n.shape)
dp_n_b      = np.zeros(tp_n.shape)
dp_n_c      = np.zeros(tp_n.shape)

# Parallelization overhead
for ncores in list_cores:
    p_n[ncores-1,:]       = tp_n[ncores-1,:] - tp_n[0,:]/ncores
    p_n_max[ncores-1]     = tp_n_max[ncores-1] - tp_n_max[0]/ncores
    p_n_min[ncores-1]     = tp_n_min[ncores-1] - tp_n_min[0]/ncores
    p_n_med[ncores-1]     = tp_n_med[ncores-1] - tp_n_med[0]/ncores
        
# Discrete approximation of P'(n) = P(n) - P(n-1) or P'(n) = P(n+1) - P(n)
for ncores in list_cores:
    if ncores > 1:
        dp_n_f[ncores-1,:]      = p_n[ncores-1,:] - p_n[ncores-2,:]
    if ncores < list_cores[num_list_cores-1]:
        dp_n_b[ncores-1,:]    = p_n[ncores,:] - p_n[ncores-1,:]


tp_n_med_norm = 100*tp_n_med/tp_n_med[0]
p_n_med_norm = 100*p_n_med/tp_n_med[0]


# Plot
####################################

fs  = 14
wid = 0.8

fig, ax = plt.subplots(figsize=(7, 5))


# Total runtime
#plt.fill_between(list_cores, tp_n_min, tp_n_max, facecolor='lightgreen')
#plt.scatter(list_cores, tp_n_med, color='darkgreen',label="Parallelized Tasks",marker='o')
p1 = plt.fill_between(list_cores, 0, tp_n_med_norm, facecolor='green',label="Parallelized Tasks")

# Parallelization overhead
#plt.fill_between(list_cores, p_n_min, p_n_max, facecolor='pink')
#plt.scatter(list_cores, p_n_med, color='red',label="Parallelization Overhead",marker='s')
p2 = plt.fill_between(list_cores, 0, p_n_med_norm, facecolor='darkorange',label="Parallelization Overhead")

#p3 = ax.axvspan(n_b_opt, n_f_opt, facecolor='0.9', alpha=0.5, edgecolor='0.1',label="Optimal Number of Cores")

# Axes labels
plt.xlabel('Number of Cores', color = 'black', fontsize=fs)
plt.ylabel('% of Total Serial Runtime', color = 'black', fontsize=fs)

# Axes ticks
plt.xticks(list_cores,color = 'black',fontsize = fs, rotation=90)
plt.yticks(np.arange(0,110,10),color = 'black',fontsize = fs)

# Background grid
plt.grid()
ax.xaxis.grid(linestyle=':')
ax.yaxis.grid(linestyle=':')

# Legend
plt.legend(loc='upper right', fontsize = fs)
plt.savefig(fig_name, dpi = 300)
plt.show()

