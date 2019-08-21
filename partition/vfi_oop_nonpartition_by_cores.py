# -*- coding: utf-8 -*-
"""
Created on Sat Sep  8 00:12:12 2018
Value Function Iteration

@author: Don Koh
"""
import json
from pprint import pprint
import numpy as np
import time
import sys
from scipy.stats import norm
from scipy.optimize import fminbound
import matplotlib.pyplot as plt
from multiprocessing import Pool

 

#--------------------------------#
#       HOUSEHOLD CLASS
#--------------------------------#
class Household:
    # INITIALIZATION OF THE CLASS WITH GIVEN PARAMETERS VALUES
    def __init__(self,  T           = 10,
                        na          = 500,
                        amin        = 0.0,
                        amax        = 10.0,
                        ne          = 15,
                        σ           = 0.2058,
                        ρ           = 0.95,
                        m           = 1.5,
                        γ           = 2,
                        β           = 0.97,
                        r           = 0.02,
                        w           = 1):
        
        # LIFE TIME
        self.T              = T
        
        # GRID FOR a
        self.na             = na
        self.amin           = amin
        self.amax           = amax
        self.agrid          = np.linspace(self.amin,self.amax,self.na)

        # GRID FOR e AND TRANSITION MATRIX: PARAMETERS FOR TAUCHEN
        self.ne             = ne
        self.σ              = σ
        self.ρ              = ρ
        self.m              = m
        self.egrid          = np.zeros(self.ne)
        self.P              = np.zeros((self.ne, self.ne))

        # PARAMETERS FOR UTILITY FUNCTION
        self.γ              = γ
        self.β              = β

        # PRICES FOR PRICE TAKERS
        self.r              = r
        self.w              = w

        # INITIALIZE VALUE FUNCTION AND POLICY FUNCTIONS
        self.V              = np.zeros((T, na, ne))
        self.a1             = np.zeros((T, na, ne))

        self.get_egrid()
        

    def get_egrid(self):
        '''
        This function constructs and returns the grid for state e
        and its transition matrix P using Tauchen (1986)
        '''
        σ_y = np.sqrt(self.σ**2 / (1 - self.ρ**2))
        estep = 2*σ_y*self.m / (self.ne-1)
        it = 0
        for i in range(self.ne):
            self.egrid[i] = (-self.m*np.sqrt(self.σ**2 / (1 - self.ρ**2)) + it*estep)
            it = it+1
        mm = self.egrid[1] - self.egrid[0]
        for j in range(self.ne):
            for k in range(self.ne):
                if (k == 0):
                    self.P[j, k] = norm.cdf((self.egrid[k] - self.ρ*self.egrid[j] + (mm/2))/self.σ)
                elif (k == self.ne-1):
                    self.P[j, k] = 1 - norm.cdf((self.egrid[k] - self.ρ*self.egrid[j] - (mm/2))/self.σ)
                else:
                    self.P[j, k] = norm.cdf((self.egrid[k] - self.ρ*self.egrid[j] + (mm/2))/self.σ) - \
                                   norm.cdf((self.egrid[k] - self.ρ*self.egrid[j] - (mm/2))/self.σ)

        # EXPONENTIAL OF THE GRID
        for i in range(self.ne):
            self.egrid[i] = np.exp(self.egrid[i])


    def util(self,cons):
        '''
        This function returns the value of CRRA utility with ssigma
        u(c) = c**(1-ssigma)/(1-ssigma)
        '''
        if self.γ != 1:
            uu = cons**(1-self.γ)/(1-self.γ)
        else:
            uu = np.log(cons)
        
        return uu
    
    def mutil(self,cons):
        '''
        This function returns the value of CRRA utility with ssigma
        u(c) = c**(1-ssigma)/(1-ssigma)
        '''
        if self.γ != 1:
            mu = cons**(-self.γ)
        else:
            mu = 1/cons
        
        return mu
    
    def set_a1(self,age,ap):
        '''
        This function updates the value of saving
        '''
        self.a1[age,:,:]   = ap

    def get_a1(self):
        '''
        This function updates the value of saving
        '''
        return self.a1

    def set_V(self,age,VV):
        '''
        This function updates the value function
        '''
        self.V[age,:,:]   = VV

    def get_V(self):
        '''
        This function updates the value function
        '''
        return self.V


class Household_sub(Household):
    def __init__(self, na_sub, agrid_sub, agrid_ind):
        
        # MODEL PARAMETERS
        Household.__init__(self)
        self.na         = na_sub
        self.agrid      = agrid_sub
        self.agrid_ind  = agrid_ind
        self.V          = np.zeros((self.T, self.na, self.ne))
        self.a1         = np.zeros((self.T, self.na, self.ne))


def partition_mst(hh_mst,num,na_sub):

    if (hh_mst.na!=num*na_sub):
        sys.stdout.write('patition_mstclass: number of grid points mismatch.\n')
    hh_list    = []
    agrid_mst   = hh_mst.agrid

    for ih in range(num):
        agrid_ind   = [i*num+ih for i in range(na_sub)]
        if ih>0:
            agrid_ind.insert(0,0)
        agrid_sub   = agrid_mst[agrid_ind]
        sub_class   = Household_sub(len(agrid_sub), agrid_sub, agrid_ind)
        hh_list.append(sub_class)        

    return hh_list



def merge_sub(hh_mst,hh_list):
    
    V_mst   = np.zeros((hh_mst.T,hh_mst.na,hh_mst.ne))
    a1_mst  = np.zeros((hh_mst.T,hh_mst.na,hh_mst.ne))

    for ih, hh_sub in enumerate(hh_list):
        agrid_ind                   = hh_sub.agrid_ind
        V_sub                       = hh_sub.get_V()
        a1_sub                      = hh_sub.get_a1()
        for age in range(hh_mst.T):
            if ih>0:
                V_mst[age,agrid_ind[1:],:]      = V_sub[age,1:,:]
                a1_mst[age,agrid_ind[1:],:]     = a1_sub[age,1:,:]
            else:
                V_mst[age,agrid_ind,:]          = V_sub[age,:,:]
                a1_mst[age,agrid_ind,:]         = a1_sub[age,:,:]
    
    for age in range(hh_mst.T):
        hh_mst.set_V(age,V_mst[age,:,:])
        hh_mst.set_a1(age,a1_mst[age,:,:])
    
    return hh_mst


#@jit
def bellman(ap,hh,a0,e0,V1,P1):
    '''
    This function computes bellman equation for a given state (a0,e0).
    Input:
        ap: evaluating point
        hh: household object
        (a0,e0) state
        V1: value function at age t+1 evaluated at (a',e')
        P1: probability distribution of e' conditional on the current e0
    Output:
        -vv: bellman equation
    '''
    ne, agrid, r, w, β = hh.ne, hh.agrid, hh.r, hh.w, hh.β
    
    # Initialize an expected continuation value
    EVa1    = 0
    
    # Compute expected continuation value for each e'
    for iep in range(ne):
        
        # Transition probability conditional on current e0, 
        eprob       = P1[iep]
        
        # Interpolate next period's value function evaluated at (a',e')
        # using 1-dimensional interpolation function in numpy
        V1a         = np.interp(ap,agrid,V1[:,iep])
        
        # Interpolated value cannot be NaN or Inf
        if np.isnan(V1a) or np.isinf(V1a): print("bellman: V1a is NaN.")
        
        # Compute expected value
        EVa1        += eprob*V1a

    # Compute consumption at a given (a0,e0) and a'       
    cons = (1 + r) * a0 + w * e0 - ap
    
    # Consumption must be non-negative
    if cons<=0:
        vv = (cons-10)*1000
    else:
        # Compute value function
        vv  = hh.util(cons) + β*EVa1
    
    return -vv


def vfi_opt(hh):
    '''
    This function returns value function for a given state (a0,e0) and age
    Input:
        hh: household object
    '''
    # Model Parameters
    T       = hh.T
    na      = hh.na
    ne      = hh.ne
    r       = hh.r
    w       = hh.w
    agrid   = hh.agrid
    egrid   = hh.egrid
    amin    = agrid[0]
    amax    = agrid[na-1]
    P       = hh.P
    V1      = np.empty((na,ne))
    
    for age in reversed(range(T)):
#        sys.stdout.write('Age = %d.\n' % age)
        
        # EMPTY BIN FOR VALUE FUNCTION AND POLICY FUNCITONS
        V_temp  = np.zeros((na,ne))
        a1_temp = np.zeros((na,ne))
        
        # NO GRID SEARCH AT AGE T
        if(age == T-1):
            for ind in range(na*ne):
                ia = ind // ne
                ie = ind % ne
                c0  = (1 + r) * agrid[ia] + w * egrid[ie]
                V_temp[ia,ie] = hh.util(c0)   # VALUE FUNCTION
    
        # GRID SEARCH AT AGE < T
        else:
            # Continuation Value
            for ind in range(na*ne):
                ia  = ind // ne
                ie  = ind % ne
                a0  = agrid[ia]
                e0  = egrid[ie]
                a1_temp[ia,ie]  = fminbound(bellman,amin,amax,args=(hh,a0,e0,V1,P[ie,:]))
                V_temp[ia,ie]   = -bellman(a1_temp[ia,ie],hh,a0,e0,V1,P[ie,:])

        # UPDATE VALUE FUNCTION AND POLICY FUNCTIONS
        hh.set_V(age,V_temp)
        hh.set_a1(age,a1_temp)

        V1  = V_temp
        
    return hh


def EE_error(hh):
    '''
    This function returns Euler equation errors
    Input:
        hh: household object
    '''
    a1      = hh.get_a1()
    agrid   = hh.agrid
    egrid   = hh.egrid
    r       = hh.r
    w       = hh.w
    T       = hh.T
    na      = hh.na
    ne      = hh.ne
    P       = hh.P
    β       = hh.β
    γ       = hh.γ
    
    cons    = np.zeros((T,na,ne))
    binding = np.zeros((T,na,ne))
    mutil   = np.zeros((T,na,ne))
    EE_error    = np.zeros((T-2,na,ne))

    for age in range(T):
        for ind in range(na*ne):
            ia = ind // ne
            ie = ind % ne
            cons[age,ia,ie]    = (1 + r) * agrid[ia] + w * egrid[ie] - a1[age,ia,ie]
            if cons[age,ia,ie]<=0 or a1[age,ia,ie]<=1e-10: binding[age,ia,ie] = 1
            else: mutil[age,ia,ie]   = hh.mutil(cons[age,ia,ie])
            
    for age in range(T-2):
        for ind in range(na*ne):
            ia = ind // ne
            ie = ind % ne
            cons0   = cons[age,ia,ie]
            if binding[age,ia,ie]!=1:
            
                # Compute expected marginal utility
                Emu    = 0
                for iep in range(ne):
                    mutil1      = np.interp(a1[age,ia,ie],agrid,mutil[age+1,:,iep])
                    Emu        += P[ie,iep]*mutil1
            
                EE_error[age,ia,ie]     = 1 - (β*(1+r)*Emu)**(-1/γ)/cons0

            else:
                EE_error[age,ia,ie]     = 0
    
    return EE_error



def run_vfi(num_core, na_sub, ne):


    '''
    This function runs the main process.
    '''
    
    s0          = time.time()
    
    # EMPTY BIN FOR RUNTIME RESULTS
    run_time    = []

    na          = num_core * na_sub
    
    # INITILIZE THE HOUSEHOLD CLASS
    hh          = Household(na=na,ne=ne)
    
    # Partition the master class into sub-classes
    hh_list     = partition_mst(hh, num_core, na_sub)
        
    # Value function iteration in parallel
    s2          = time.time()
    hh_list     = vfi_opt(hh_list)
    f2          = time.time() - s2

    # Merge sub-classes
    hh          = merge_sub(hh, hh_list)

        
    # TOTAL RUNTIME
    f0 = time.time() - s0
    run_time = [f0,f2]
#    sys.stdout.write('TOTAL LIFECYCLE RUNTIME:  = %f seconds.\n' % round(f0_sum, 4))
#    sys.stdout.write('TOTAL ELAPSED TIME:  = %f seconds.\n' % round(f0, 4))

    error = EE_error(hh)
    
    return run_time, error


    
    
    
    

#---------------------------------------------------------------------#
#---------------------------------------------------------------------#
#---------------------------------------------------------------------#
#                       START COMPUTING!!!                            # 
#---------------------------------------------------------------------#
#---------------------------------------------------------------------#
#---------------------------------------------------------------------#
if __name__ == "__main__":
    # Number of cores
    num_cores       = int(sys.argv[1])
    na_sub          = int(sys.argv[2])
#    num_cores       = 1
#    na_sub          = 100
    na              = num_cores * na_sub
    
    # Run time memory    
    T           = 10
    nrun        = 10
    ne          = 15
    
    # Iterate on the number of cores
    # Store the results as JSON data
    data = []
    EEerr = []
    for num_runs in range(nrun):
        
        results, error = run_vfi(num_cores, na_sub, ne)
        
        # Store the total run time and initialization in a JSON data
        data.append({
                'core': int(num_cores),
                'run': int(num_runs),
                'age': int(0),
                'runtime': results[0]
        })
        data.append({
                'core': int(num_cores),
                'run': int(num_runs),
                'age': int(1),
                'runtime': results[1]
        })
        for age in range(T-2):
            for ind in range(na*ne):
                ia = ind // ne
                ie = ind % ne

                EEerr.append({
                        'core': int(num_cores),
                        'run': int(num_runs),
                        'age': age,
                        'a_state': ia,
                        'e_state': ie,
                        'error': error[age,ia,ie]
                })
            
        # Print out to a console window
        print("####### ", num_runs+1, " times run with ",num_cores, " cores #######\n")
        print ("Total Elapsed Time: ", round(results[0], 4), " seconds.")
                
        print(" ")    
    
    file_name = "runtime_by_cores_" + str(num_cores) + "_" + str(na_sub) + ".txt"
    # Write the data to a output file
    with open(file_name,"w+") as outfile:
        json.dump(data, outfile)
        
    # Retrieve the results from the output file
    with open(file_name) as json_data:
        d = json.load(json_data)
        pprint(d)
    for ik in range(len(d)):
        print(d[ik]['core'],d[ik]['run'],d[ik]['age'],d[ik]['runtime'])

    file_name1 = "error_by_cores_" + str(num_cores) + "_" + str(na_sub) + ".txt"
    # Write the data to a output file
    with open(file_name1,"w+") as outfile1:
        json.dump(EEerr, outfile1)

    