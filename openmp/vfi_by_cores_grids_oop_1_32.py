# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 12:14:31 2018

@author: dkoh
"""

import json
from pprint import pprint
import numpy as np
import time
from scipy.stats import norm
from scipy.optimize import fminbound
from multiprocessing import Pool
from functools import partial
 

#--------------------------------#
#       HOUSEHOLD CLASS
#--------------------------------#
class Household:
    # INITIALIZATION OF THE CLASS WITH GIVEN PARAMETERS VALUES
    def __init__(self,  T           = 10,
                        na          = 1000,
                        amin        = 0.0,
                        amax        = 10.0,
                        ne          = 15,
                        ssigma_eps  = 0.02058,
                        rho_eps     = 0.99,
                        m           = 1.5,
                        ssigma      = 2,
                        bbeta       = 0.97,
                        r           = 0.07,
                        w           = 5):

        # LIFE TIME
        self.T              = T
        
        # GRID FOR a
        self.na             = na
        self.amin           = amin
        self.amax           = amax
        self.agrid          = np.zeros(self.na)

        # GRID FOR e AND TRANSITION MATRIX: PARAMETERS FOR TAUCHEN
        self.ne             = ne
        self.ssigma_eps     = ssigma_eps
        self.rho_eps        = rho_eps
        self.m              = m
        self.egrid          = np.zeros(self.ne)
        self.P              = np.zeros((self.ne, self.ne))

        # PARAMETERS FOR UTILITY FUNCTION
        self.ssigma         = ssigma
        self.bbeta          = bbeta

        # PRICES FOR PRICE TAKERS
        self.r              = r
        self.w              = w

        # INITIALIZE VALUE FUNCTION AND POLICY FUNCTIONS
        self.V              = np.zeros((T, na, ne))
        self.c0             = np.zeros((T, na, ne))
        self.a1             = np.zeros((T, na, ne))


    def get_agrid(self):
        '''
        This function constructs and returns the grid for state a.
        The grid points are in equidistance.
        '''
        astep = (self.amax - self.amin) /(self.na - 1)
        it = 0
        for i in range(self.na):
            self.agrid[i] = self.amin + it*astep
            it = it+1        
        return self.agrid

    def get_egrid(self):
        '''
        This function constructs and returns the grid for state e
        and its transition matrix P using Tauchen (1986)
        '''
        ssigma_y = np.sqrt(self.ssigma_eps**2 / (1 - self.rho_eps**2))
        estep = 2*ssigma_y*self.m / (self.ne-1)
        it = 0
        for i in range(self.ne):
            self.egrid[i] = (-self.m*np.sqrt(self.ssigma_eps**2 / (1 - self.rho_eps**2)) + it*estep)
            it = it+1
        mm = self.egrid[1] - self.egrid[0]
        for j in range(self.ne):
            for k in range(self.ne):
                if (k == 0):
                    self.P[j, k] = norm.cdf((self.egrid[k] - self.rho_eps*self.egrid[j] + (mm/2))/self.ssigma_eps)
                elif (k == self.ne-1):
                    self.P[j, k] = 1 - norm.cdf((self.egrid[k] - self.rho_eps*self.egrid[j] - (mm/2))/self.ssigma_eps)
                else:
                    self.P[j, k] = norm.cdf((self.egrid[k] - self.rho_eps*self.egrid[j] + (mm/2))/self.ssigma_eps) - \
                                   norm.cdf((self.egrid[k] - self.rho_eps*self.egrid[j] - (mm/2))/self.ssigma_eps)

        # EXPONENTIAL OF THE GRID
        for i in range(self.ne):
            self.egrid[i] = np.exp(self.egrid[i])

        return self.egrid

    def util(self,cons):
        '''
        This function returns the value of CRRA utility with ssigma
        u(c) = c**(1-ssigma)/(1-ssigma)
        '''
        if self.ssigma != 1:
            uu = cons**(1-self.ssigma)/(1-self.ssigma)
        else:
            uu = np.log(cons)
        
        return uu
    
    def set_c0(self,age,ia,ie,cons):
        '''
        This function updates the value of consumption
        '''
        self.c0[age,ia,ie]   = cons
    
    def set_a1(self,age,ia,ie,ap):
        '''
        This function updates the value of saving
        '''
        self.a1[age,ia,ie]   = ap

    def set_V0(self,age,ia,ie,VV):
        '''
        This function updates the value function
        '''
        self.V[age,ia,ie]   = VV


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
    # Initialize an expected continuation value
    EVa1    = 0
    
    # Compute expected continuation value for each e'
    for iep in range(hh.ne):
        
        # Transition probability conditional on current e0, 
        eprob       = P1[iep]
        
        # Interpolate next period's value function evaluated at (a',e')
        # using 1-dimensional interpolation function in numpy
        V1a         = np.interp(ap,hh.agrid,V1[:,iep])
        
        # Interpolated value cannot be NaN or Inf
        if np.isnan(V1a) or np.isinf(V1a): print("bellman: V1a is NaN.")
        
        # Compute expected value
        EVa1        += eprob*V1a

    # Compute consumption at a given (a0,e0) and a'       
    cons = (1 + hh.r)*a0 + hh.w*e0 - ap
    
    # Consumption must be non-negative
    if cons<=0:
        vv = -1e10
    else:
        # Compute value function
        vv  = hh.util(cons) + hh.bbeta*EVa1
    
    return -vv


def vfi_opt(hh,age,ind):
    '''
    This function returns value function for a given state (a0,e0) and age
    Input:
        hh: household object
        ind: a unique state that corresponds to a pair of state (a0,e0)
        age: age
    ''' 
    # Index of current state (a0,e0)
    ia = ind // hh.ne
    ie = ind % hh.ne
    
    # Current state (a0,e0)
    a0 = hh.agrid[ia]
    e0 = hh.egrid[ie]

    # At each state (a0,e0), bounded minimization of a function w.r.t. a' over [amin,amax]
    # using fminbound function in scipy.
    aa = fminbound(bellman,hh.amin,hh.amax,args=(hh,a0,e0,hh.V[age+1,:,:],hh.P[ie,:]))
    VV = -bellman(aa,hh,a0,e0,hh.V[age+1,:,:],hh.P[ie,:])
    #### full_output option takes longer than recalculating value function
#    aa, VV, err, n_func = fminbound(bellman,hh.amin,hh.amax,args=(hh,a0,e0,hh.V[age+1,:,:],hh.P[ie,:]),full_output=True)
    cc = (1 + hh.r)*a0 + hh.w*e0 - aa

    return VV,cc,aa


def run_vfi_opt(num_cores,na):
    '''
    This function runs the main process.
    ''' 
    
    s0          = time.time()
    f0_sum      = 0
    
    #------------------------------------------#
    #      STEP1: INITIALIZATION
    #------------------------------------------#
#    print("INITIALIZATION: ")
    # NUMBER OF CORES TO BE USED
#    print("   Number of Cores = ", num_cores)
#    print(" ")
    
    
    # INITILIZE THE HOUSEHOLD CLASS
    hh          = Household(na=na)
    
    # GRID CONSTRUCTION FOR a
    agrid       = hh.get_agrid()
    
    # GRID CONSTRUCTION FOR e
    egrid       = hh.get_egrid()
    
    
    #------------------------------------------#
    #     STEP2: LIFECYCLE COMPUTATION
    #------------------------------------------#
#    print("LIFECYCLE COMPUTATION: ")
#    print(" ")
    
    # EMPTY BIN FOR RUNTIME RESULTS
    try:
        run_time    = np.zeros(hh.T)
        pool = Pool(processes=num_cores)
        for age in reversed(range(hh.T)):
            s2  = time.time()
        
            # EMPTY BIN FOR VALUE FUNCTION AND POLICY FUNCITONS
            results = np.zeros((hh.na*hh.ne,3))
            
            # NO GRID SEARCH AT AGE T
            if(age == hh.T-1):
                for ind in range(hh.na*hh.ne):
                    ia = ind // hh.ne
                    ie = ind % hh.ne
                    cc              = (1 + hh.r)*agrid[ia] + hh.w*egrid[ie]
                    if cc<=0: cc = 1e-5
                    results[ind][0] = hh.util(cc)   # VALUE FUNCTION
                    results[ind][1] = cc            # CONSUMPTION
                    results[ind][2] = 0.0           # SAVING
        
            # GRID SEARCH AT AGE < T
            else:
                # Parallel grid search
                vfi_opt_partial = partial(vfi_opt, hh, age)
                results = pool.map(vfi_opt_partial, np.arange(hh.na*hh.ne))
        
                # UPDATE VALUE FUNCTION AND POLICY FUNCTIONS
            for ind in range(hh.na*hh.ne):
                ia = ind // hh.ne
                ie = ind % hh.ne
                hh.set_V0(age,ia,ie,results[ind][0])
                hh.set_c0(age,ia,ie,results[ind][1])
                hh.set_a1(age,ia,ie,results[ind][2])
        
        
            f2 = time.time() - s2
            run_time[age] = f2
            f0_sum  += f2
    #        print( "Age: ", age+1, ". Time: ", round(f2, 4), " seconds.")
    finally:
        pool.close()
        pool.join()
    
    # TOTAL RUNTIME
    f0 = time.time() - s0
    run_time = np.insert(run_time, 0, [f0, f0_sum])
#    print ("TOTAL ELAPSED TIME: ", round(f0_sum, 4), " seconds. \n")

    return run_time


if __name__ == "__main__":
    
    if len(sys.argv) > 1:
        num_core = int(sys.argv[1])
    
    # Number of cores
#    list_cores      = np.arange(1,17,1)
#    list_cores      = np.arange(2,9,1)
    list_cores      = [num_core]
    num_list_cores  = len(list_cores)
    
    # Number of cores
#    list_grids      = np.arange(100,501,100)
    list_grids      = np.arange(500,1501,100)
    num_list_grids  = len(list_grids)
    
    # Number of runs
    nrun        = 3
    
    data = []
    # Iterate on the number of cores
    for num_cores in list_cores:
        for na in list_grids:
            for num_runs in range(nrun):
                
                results  = run_vfi_opt(num_cores, na)
            
                data.append({
                    'core': int(num_cores),
                    'grid': int(na),
                    'run': int(num_runs),
                    'age': int(-1),
                    'runtime': results[0]
                })
                data.append({
                    'core': int(num_cores),
                    'grid': int(na),
                    'run': int(num_runs),
                    'age': int(0),
                    'runtime': results[1]
                })
    
                # Print out to a console window
                print("####### ", num_runs+1, " times run with ",num_cores, " cores and ", na, "grids  #######")
                print ("Total Elapsed Time: ", round(results[1], 4), " seconds.")
                print(" ")
    
    
    
    # Write the data to a output file
    file_name = "runtime_vfi_by_cores_grids_oop_1_32_" + str(list_cores[0]) + ".txt"    
    with open(file_name,"w+") as outfile:
        json.dump(data, outfile)
    
    
    # Retrieve the results from the output file
    with open(file_name) as json_data:
        d = json.load(json_data)
        pprint(d)
    for ik in range(len(d)):
        print(d[ik]['core'],d[ik]['grid'],d[ik]['run'],d[ik]['age'],d[ik]['runtime'])
        
