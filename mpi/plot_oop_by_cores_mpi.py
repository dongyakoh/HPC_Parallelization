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






def plot_design(plt, height):
    
    plt.annotate("", xy=(1, height), xycoords='data',
                 xytext=(32, height), textcoords='data',
                 arrowprops=dict(arrowstyle="<->", connectionstyle="arc3"))
    plt.annotate("", xy=(32, height), xycoords='data',
                 xytext=(60, height), textcoords='data',
                 arrowprops=dict(arrowstyle="<->", connectionstyle="arc3"))
    plt.text(16, height, "1 Node",
                 {'color': 'k', 'fontsize': 12, 'ha': 'center', 'va': 'center',
                  'bbox': dict(boxstyle="round", fc="w", ec="k", pad=0.2)})
    plt.text(48, height, "2 Nodes",
                 {'color': 'k', 'fontsize': 12, 'ha': 'center', 'va': 'center',
                  'bbox': dict(boxstyle="round", fc="w", ec="k", pad=0.2)})
    plt.axvline(x=32,color='k',linestyle = '-')
        
    plt.tight_layout()



max_core    = 32
file_name0  = "runtime_oop_by_cores_mpi_"
list_cores  = [1,2,4,8,12,15,20,25,30,40,50,60]
plot_list   = [2,4,8,12,15,20,25,30,40,50,60]
    
num_list_cores  = len(list_cores)
nrun        = 10

tp_n_temp   = np.zeros(nrun)
tp_n_max    = np.zeros(num_list_cores)
tp_n_min    = np.zeros(num_list_cores)
tp_n_med    = np.zeros(num_list_cores)


for idx_core, num_core in enumerate(list_cores):

    file_name = "runtime_oop_by_cores_mpi_" + str(num_core) + ".txt"

    with open(file_name) as json_data:
        d = json.load(json_data)

    # Read data from file
    for ik in range(len(d)):
        if d[ik]['age']==0 and d[ik]['run']>=0 and d[ik]['core']>0:
            tp_n_temp[d[ik]['run']]   = d[ik]['runtime']


    # Total runtime of parallelizable code
    tp_n_max[idx_core]    = np.amax(tp_n_temp)
    tp_n_min[idx_core]    = np.amin(tp_n_temp)
    tp_n_med[idx_core]    = np.median(tp_n_temp)

 

# Parallelization overhead
p_n_max     = np.zeros(num_list_cores)
p_n_min     = np.zeros(num_list_cores)
p_n_med     = np.zeros(num_list_cores)
for idx_core, num_core in enumerate(list_cores):
    p_n_max[idx_core]     = tp_n_max[idx_core] - tp_n_max[0]/num_core
    p_n_min[idx_core]     = tp_n_min[idx_core] - tp_n_min[0]/num_core
    p_n_med[idx_core]     = tp_n_med[idx_core] - tp_n_med[0]/num_core
        

# Speed Up
sp_n_max     = np.zeros(num_list_cores)
sp_n_min     = np.zeros(num_list_cores)
sp_n_med     = np.zeros(num_list_cores)
for idx_core, num_core in enumerate(list_cores):
    sp_n_max[idx_core]     = tp_n_max[0]/tp_n_max[idx_core]
    sp_n_min[idx_core]     = tp_n_min[0]/tp_n_min[idx_core]
    sp_n_med[idx_core]     = tp_n_med[0]/tp_n_med[idx_core]

# Efficiency
ef_n_max     = np.zeros(num_list_cores)
ef_n_min     = np.zeros(num_list_cores)
ef_n_med     = np.zeros(num_list_cores)
for idx_core, num_core in enumerate(list_cores):
    ef_n_max[idx_core]     = tp_n_max[0]/(tp_n_max[idx_core]*num_core)
    ef_n_min[idx_core]     = tp_n_min[0]/(tp_n_min[idx_core]*num_core)
    ef_n_med[idx_core]     = tp_n_med[0]/(tp_n_med[idx_core]*num_core)   

tp_n_med_norm = 100*tp_n_med/tp_n_med[0]
p_n_med_norm = 100*p_n_med/tp_n_med[0]


# Plot
####################################

fs  = 14
wid = 0.8

#plt.rc('text', usetex=True)
fig1, ax1 = plt.subplots(figsize=(7, 5))

p1 = plt.plot(list_cores[1:], p_n_med_norm[1:], '-o')
plt.xlabel('# of Cores', color = 'black', fontsize=fs)
plt.ylabel('% of Total Serial Runtime', color = 'black', fontsize=fs)
plt.xticks(plot_list,color = 'black',fontsize = fs, rotation=90)
plt.yticks(np.arange(0,25.01,2),color = 'black',fontsize = fs)
plt.grid()
ax1.xaxis.grid(linestyle=':')
ax1.yaxis.grid(linestyle=':')
plot_design(plt,20.0)
fig_name = "po_by_cores_mpi.png"
plt.savefig(fig_name, dpi = 300, bbox_inches = "tight")
plt.show()


fig2, ax2 = plt.subplots(figsize=(7, 5))
p1 = plt.plot(list_cores[1:], sp_n_med[1:], '-o')
p2 = plt.plot(list_cores[1:], list_cores[1:],'--',color = 'gray',label='45$^{\circ}$ line')
plt.xlabel('# of Cores', color = 'black', fontsize=fs)
plt.ylabel('Relative Performance', color = 'black', fontsize=fs)
plt.xticks(plot_list,color = 'black',fontsize = fs, rotation=90)
plt.yticks(np.arange(0,61,10),color = 'black',fontsize = fs)
plt.grid()
ax2.xaxis.grid(linestyle=':')
ax2.yaxis.grid(linestyle=':')
plot_design(plt,50)
plt.text(56, 58, "45$^{\circ}$",{'color': 'k', 'fontsize': 12, 'ha': 'center', 'va': 'center'})
#plt.legend(loc='lower right', fontsize = fs)
fig_name = "speedup_by_cores_mpi.png"
plt.savefig(fig_name, dpi = 300, bbox_inches = "tight")
plt.show()


"""
fig3, ax3 = plt.subplots(figsize=(7, 5))
p1 = plt.plot(list_cores[1:], ef_n_med[1:], '-o')
plt.xlabel('Number of Cores', color = 'black', fontsize=fs)
plt.ylabel('Efficiency', color = 'black', fontsize=fs)
plt.xticks(plot_list,color = 'black',fontsize = fs, rotation=90)
plt.yticks(np.arange(0.6,0.71,0.01),color = 'black',fontsize = fs)
plt.grid()
ax3.xaxis.grid(linestyle=':')
ax3.yaxis.grid(linestyle=':')
plot_design(plt,0.68)
fig_name = "efficiency_by_cores_mpi.png"
plt.savefig(fig_name, dpi = 300, bbox_inches = "tight")
plt.show()
"""
