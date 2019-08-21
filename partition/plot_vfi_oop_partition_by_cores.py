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



max_core    = 32
na_sub      = 100
nrun        = 10

core_list   = np.arange(2,max_core+1,1)
na_list     = np.arange(200,max_core*na_sub+1,100)


# Fixing 1 core
tp_core0    = np.zeros((len(na_list),nrun))
for j,na in enumerate(na_list):
        
    file_name0 = "runtime_by_cores_1_" + str(na) + ".txt"
    with open(file_name0) as json_data:
        d0 = json.load(json_data)

    for ik in range(len(d0)):
        if d0[ik]['age']==0:
            tp_core0[j,d0[ik]['run']] = d0[ik]['runtime']




# Fixing 100 na grid points
tp_na0      = np.zeros((len(core_list),nrun))
for i,core in enumerate(core_list):

    file_name1 = "runtime_by_cores_" + str(core) + "_100.txt"
    with open(file_name1) as json_data:
        d1 = json.load(json_data)

    for ik in range(len(d1)):
        if d1[ik]['age']==0:
            tp_na0[i,d1[ik]['run']] = d1[ik]['runtime']


# Compute parallelization overhead
tp_na0_norm     = np.zeros((len(core_list),nrun))
p_na0           = np.zeros((len(core_list),nrun))
p_na0_norm      = np.zeros((len(core_list),nrun))
ef_na0          = np.zeros((len(core_list),nrun))
sp_na0          = np.zeros((len(core_list),nrun))
for i,core in enumerate(core_list):
    p_na0[i,:]          = tp_na0[i,:] - tp_core0[i,:]/core
    sp_na0[i,:]         = tp_core0[i,:]/tp_na0[i,:]
    ef_na0[i,:]         = tp_core0[i,:]/(tp_na0[i,:]/core)
    tp_na0_norm[i,:]    = 100*tp_na0[i,:]/tp_core0[i,:]
    p_na0_norm[i,:]     = 100*p_na0[i,:]/tp_core0[i,:]


tp_core0_med        = np.median(tp_core0,axis=1)
tp_na0_med          = np.median(tp_na0,axis=1)
p_na0_med           = np.median(p_na0,axis=1)
sp_na0_med          = np.median(sp_na0,axis=1)
ef_na0_med          = np.median(ef_na0,axis=1)
tp_na0_norm_med     = np.median(tp_na0_norm,axis=1)
p_na0_norm_med      = np.median(p_na0_norm,axis=1)



# Plot
####################################

fs  = 14
wid = 0.8



# Parallelization Overhead
fig, ax0 = plt.subplots(figsize=(7, 5))
p1 = plt.plot(core_list, p_na0_norm_med, '-^',label="Partitioned")
plt.xlabel('# of Cores', color = 'black', fontsize=fs)
plt.ylabel('% of Total Serial Runtime', color = 'black', fontsize=fs)
plt.xticks(core_list,color = 'black',fontsize = fs, rotation=90)
plt.yticks(np.arange(-2.5,0.1,0.5),color = 'black',fontsize = fs)
plt.grid()
ax0.xaxis.grid(linestyle=':')
ax0.yaxis.grid(linestyle=':')
#plt.legend(loc='upper left', fontsize = fs)
fig_name = "po_partition_by_cores.png"
plt.savefig(fig_name, dpi = 300,bbox_inches='tight')
plt.show()


# Speed Up
fig, ax1 = plt.subplots(figsize=(7, 5))
p1 = plt.plot(core_list, sp_na0_med, '-o')
p2 = plt.plot(core_list, core_list, '--',color = 'gray')
plt.xlabel('# of Cores', color = 'black', fontsize=fs)
plt.ylabel('Relative Performance', color = 'black', fontsize=fs)
plt.xticks(core_list,color = 'black',fontsize = fs, rotation=90)
plt.yticks(np.arange(0,61,10),color = 'black',fontsize = fs)
plt.grid()
ax1.xaxis.grid(linestyle=':')
ax1.yaxis.grid(linestyle=':')
plt.text(31, 33, "45$^{\circ}$",{'color': 'k', 'fontsize': 12, 'ha': 'center', 'va': 'center'})
#lplt.legend(loc='upper left', fontsize = fs)
fig_name = "speed_partition_by_cores.png"
plt.savefig(fig_name, dpi = 300,bbox_inches='tight')
plt.show()

"""
# Efficiency
fig, ax2 = plt.subplots(figsize=(7, 5))
p1 = plt.plot(core_list, ef_na0_med, '-o',label="Partitioned")
plt.xlabel('# of Cores', color = 'black', fontsize=fs)
plt.ylabel('Efficiency', color = 'black', fontsize=fs)
plt.xticks(core_list,color = 'black',fontsize = fs, rotation=90)
#plt.yticks(np.arange(2,17,1),color = 'black',fontsize = fs)
plt.grid()
ax2.xaxis.grid(linestyle=':')
ax2.yaxis.grid(linestyle=':')
plt.legend(loc='upper left', fontsize = fs)
fig_name = "efficiency_partition_by_cores.png"
plt.savefig(fig_name, dpi = 300)
plt.show()


# One core
fig, ax3 = plt.subplots(figsize=(7, 5))
tp_core0_med_norm = tp_core0_med/tp_core0_med[0]
p1 = plt.plot(na_list, tp_core0_med_norm, '-o',label="Partitioned")
p2 = plt.plot(na_list, core_list, '--')
plt.xlabel('Number of Cores', color = 'black', fontsize=fs)
#plt.ylabel('Efficiency', color = 'black', fontsize=fs)
plt.xticks(na_list,color = 'black',fontsize = fs, rotation=90)
#plt.yticks(y_label,color = 'black',fontsize = fs)
plt.grid()
ax3.xaxis.grid(linestyle=':')
ax3.yaxis.grid(linestyle=':')
plt.legend(loc='upper left', fontsize = fs)
#fig_name = "efficiency_partition_by_cores.png"
#plt.savefig(fig_name, dpi = 300)
plt.show()
"""
