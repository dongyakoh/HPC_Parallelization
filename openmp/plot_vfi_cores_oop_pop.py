# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 14:12:42 2018

@author: dkoh
"""


import matplotlib.pyplot as plt
plt.rcdefaults()
plt.style.use('seaborn-bright')
import numpy as np
import json
from scipy.optimize import bisect



q_flag      = 1     # 0: med16core, 1: onenode16core, 2: mem768GB32core

if q_flag is 0:
    q_var = "_med16core"
    max_core    = 16
elif q_flag is 1:
    q_var = "_mem768GB32core"
    max_core    = 32

    
file_name0 = "runtime_vfi_by_cores_pop" + q_var + ".txt"
file_name1 = "runtime_vfi_by_cores_oop" + q_var + ".txt"
fig_name = "vfi_by_cores_ratio" + "_1_" + str(max_core) + ".png"

list_cores      = np.arange(1,max_core+1,1)
num_list_cores  = len(list_cores)


nrun        = 21
#nrun        = 2

# Read data from file
with open(file_name0) as json_data:
    d0 = json.load(json_data)
with open(file_name1) as json_data:
    d1 = json.load(json_data)


x_cores0     = np.zeros((num_list_cores,nrun))
tp_n0        = np.zeros((num_list_cores,nrun))
index = 0
for ik in range(len(d0)):
    if d0[ik]['age']==0 and d0[ik]['core']>0:
        x_cores0[index,d0[ik]['run']] = d0[ik]['core']
        tp_n0[index,d0[ik]['run']] = d0[ik]['runtime']
        if d0[ik]['run']==nrun-1:
            index += 1

x_cores1     = np.zeros((num_list_cores,nrun))
tp_n1        = np.zeros((num_list_cores,nrun))
index = 0
for ik in range(len(d1)):
    if d1[ik]['age']==0 and d1[ik]['core']>0:
        x_cores1[index,d1[ik]['run']] = d1[ik]['core']
        tp_n1[index,d1[ik]['run']] = d1[ik]['runtime']
        if d1[ik]['run']==nrun-1:
            index += 1


# Total runtime of parallelizable code
tp_n0_med    = np.median(tp_n0,axis=1)
tp_n1_med    = np.median(tp_n1,axis=1)
p_n0_med     = np.zeros(tp_n0_med.shape)
p_n1_med     = np.zeros(tp_n1_med.shape)

# Parallelization overhead
for idx_cores, ncores in enumerate(list_cores):
    p_n0_med[idx_cores]     = tp_n0_med[ncores-1] - tp_n0_med[0]/ncores
    p_n1_med[idx_cores]     = tp_n1_med[ncores-1] - tp_n1_med[0]/ncores

tp_n0_med_norm = tp_n0_med/tp_n0_med[0]
p_n0_med_norm = 100*p_n0_med/tp_n0_med[0]
tp_n1_med_norm = tp_n1_med/tp_n1_med[0]
p_n1_med_norm = 100*p_n1_med/tp_n1_med[0]


# Plot
####################################

fs  = 14
wid = 0.8

fig, ax = plt.subplots(figsize=(7, 5))


# Total runtime
#plt.fill_between(list_cores, tp_n_min, tp_n_max, facecolor='lightgreen')
#plt.scatter(list_cores, tp_n_med, color='darkgreen',label="Parallelized Tasks",marker='o')
p1 = plt.plot(list_cores[1:], p_n0_med_norm[1:], '-o',label="Variables")
p2 = plt.plot(list_cores[1:], p_n1_med_norm[1:], '-^',label="Object")


# Axes labels
plt.xlabel('Number of Cores', color = 'black', fontsize=fs)
plt.ylabel('% of Total Serial Runtime', color = 'black', fontsize=fs)

# Axes ticks
plt.xticks(list_cores[1:],color = 'black',fontsize = fs, rotation=90)
plt.yticks(np.arange(0.25,2.26,0.25),color = 'black',fontsize = fs)

# Background grid
plt.grid()
ax.xaxis.grid(linestyle=':')
ax.yaxis.grid(linestyle=':')

# Legend
plt.legend(loc='upper left', fontsize = fs)
plt.savefig(fig_name, dpi = 300)
plt.show()

