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
ne          = 15

max_error_by_core0  = np.zeros((len(na_list),nrun))
med_error_by_core0  = np.zeros((len(na_list),nrun))
max_error_by_na0    = np.zeros((len(na_list),nrun))
med_error_by_na0    = np.zeros((len(na_list),nrun))


for j,na in enumerate(na_list):
    error_core0    = np.zeros((na,ne,nrun))
    file_name0 = "error_by_cores_1_" + str(na) + ".txt"
    with open(file_name0) as json_data:
        d0 = json.load(json_data)

    for ik in range(len(d0)):
        error_core0[d0[ik]['a_state'],d0[ik]['e_state'],d0[ik]['run']] = d0[ik]['error']


    error_na0       = np.zeros((na,ne,nrun))
    ncores          = int(na/100)
    file_name1      = "error_by_cores_" + str(ncores) + "_100.txt"
    with open(file_name1) as json_data:
        d1 = json.load(json_data)

    for ik in range(len(d1)):
        error_na0[d1[ik]['a_state'],d1[ik]['e_state'],d1[ik]['run']] = d1[ik]['error']

    # Max error difference between single core and multiple cores
    for i in range(nrun):
        max_error_by_core0[j,i] = np.log10(np.max(abs(error_core0[:,:,i])))

        med_error_by_core0[j,i] = np.log10(np.median(abs(error_core0[:,:,i])))

        max_error_by_na0[j,i]   = np.log10(np.max(abs(error_na0[:,:,i])))

        med_error_by_na0[j,i]   = np.log10(np.median(abs(error_na0[:,:,i])))


max_error_by_core0_med  = np.median(max_error_by_core0,axis=1)
med_error_by_core0_med  = np.median(med_error_by_core0,axis=1)
max_error_by_na0_med    = np.median(max_error_by_na0,axis=1)
med_error_by_na0_med    = np.median(med_error_by_na0,axis=1)



# Plot
####################################

fs  = 14
wid = 0.8

fig, ax = plt.subplots(figsize=(7, 5))
p1 = plt.plot(na_list, med_error_by_na0_med, '-o',label="Partitioned")
p2 = plt.plot(na_list, med_error_by_core0_med, '-^',label="Non-partitioned")
plt.xlabel('# of Grids (= # of Cores x 100 Grids)', color = 'black', fontsize=fs)
plt.ylabel('Log10(Abs(Euler Equation Error))', color = 'black', fontsize=fs)
plt.xticks(na_list,color = 'black',fontsize = fs, rotation=90)
plt.yticks(np.arange(-4.0,-1.9,0.2),color = 'black',fontsize = fs)
plt.grid()
ax.xaxis.grid(linestyle=':')
ax.yaxis.grid(linestyle=':')
plt.legend(loc='lower left', fontsize = fs)
fig_name = "error_partition_by_cores.png"
plt.savefig(fig_name, dpi = 300,bbox_inches='tight')
plt.show()
