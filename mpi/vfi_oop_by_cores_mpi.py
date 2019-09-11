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
from scipy.stats import norm
from scipy.optimize import fminbound
from mpi4py import MPI
import sys
 

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
                        sig           = 0.2058,
                        rho           = 0.95,
                        m           = 1.5,
                        gam           = 2,
                        beta           = 0.97,
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
        self.sig              = sig
        self.rho              = rho
        self.m              = m
        self.egrid          = np.zeros(self.ne)
        self.P              = np.zeros((self.ne, self.ne))

        # PARAMETERS FOR UTILITY FUNCTION
        self.gam              = gam
        self.beta              = beta

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
        sig_y = np.sqrt(self.sig**2 / (1 - self.rho**2))
        estep = 2*sig_y*self.m / (self.ne-1)
        it = 0
        for i in range(self.ne):
            self.egrid[i] = (-self.m*np.sqrt(self.sig**2 / (1 - self.rho**2)) + it*estep)
            it = it+1
        mm = self.egrid[1] - self.egrid[0]
        for j in range(self.ne):
            for k in range(self.ne):
                if (k == 0):
                    self.P[j, k] = norm.cdf((self.egrid[k] - self.rho*self.egrid[j] + (mm/2))/self.sig)
                elif (k == self.ne-1):
                    self.P[j, k] = 1 - norm.cdf((self.egrid[k] - self.rho*self.egrid[j] - (mm/2))/self.sig)
                else:
                    self.P[j, k] = norm.cdf((self.egrid[k] - self.rho*self.egrid[j] + (mm/2))/self.sig) - \
                                   norm.cdf((self.egrid[k] - self.rho*self.egrid[j] - (mm/2))/self.sig)

        # EXPONENTIAL OF THE GRID
        for i in range(self.ne):
            self.egrid[i] = np.exp(self.egrid[i])


    def util(self,cons):
        '''
        This function returns the value of CRRA utility with ssigma
        u(c) = c**(1-ssigma)/(1-ssigma)
        '''
        if self.gam != 1:
            uu = cons**(1-self.gam)/(1-self.gam)
        else:
            uu = np.log(cons)
        
        return uu
    
    def mutil(self,cons):
        '''
        This function returns the value of CRRA utility with ssigma
        u(c) = c**(1-ssigma)/(1-ssigma)
        '''
        if self.gam != 1:
            mu = cons**(-self.gam)
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
    ne, agrid, r, w, beta = hh.ne, hh.agrid, hh.r, hh.w, hh.beta
    
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
        vv = -1e10
    else:
        # Compute value function
        vv  = hh.util(cons) + beta*EVa1
    
    return -vv


def vfi_opt(hh,age,ind):
    '''
    This function returns value function for a given state (a0,e0) and age
    Input:
        hh: household object
        ind: a unique state that corresponds to a pair of state (a0,e0)
        age: age
    ''' 

    # Model Parameters
    na      = hh.na
    ne      = hh.ne
    agrid   = hh.agrid
    egrid   = hh.egrid
    P       = hh.P
    V1      = hh.get_V()

    # Index of current state (a0,e0)
    ia = ind // ne
    ie = ind % ne
    
    # Current state (a0,e0)
    a0 = agrid[ia]
    e0 = egrid[ie]

    # At each state (a0,e0), bounded minimization of a function w.r.t. a' over [amin,amax]
    # using fminbound function in scipy.
    aa = fminbound(bellman,agrid[0],agrid[na-1],args=(hh,a0,e0,V1[age+1,:,:],P[ie,:]))
    VV = -bellman(aa,hh,a0,e0,V1[age+1,:,:],P[ie,:])

    return VV,aa


def run_vfi(comm):
    '''
    This function runs the main process.
    ''' 
    
    s0          = MPI.Wtime()
    f0_sum      = 0
    
    #------------------------------------------#
    #      STEP1: INITIALIZATION
    #------------------------------------------#
    sys.stdout.write("Running at %d of %d on %s.\n" % (comm.rank, comm.size, MPI.Get_processor_name()))    
    
    # INITILIZE THE HOUSEHOLD CLASS
    hh          = Household()


    #------------------------------------------#
    #     STEP2: LIFECYCLE COMPUTATION
    #------------------------------------------#
    for age in reversed(range(hh.T)):
        
        s2          = MPI.Wtime()

        # EMPTY BIN FOR VALUE FUNCTION AND POLICY FUNCITONS
        results = np.zeros((hh.na*hh.ne,2))
        V_temp = np.zeros((hh.na,hh.ne))
        a1_temp = np.zeros((hh.na,hh.ne))

        # NO GRID SEARCH AT AGE T
        if(age == hh.T-1):
            if comm.rank==0:
                for ind in range(hh.na*hh.ne):
                    ia = ind // hh.ne
                    ie = ind % hh.ne
                    cc              = (1 + hh.r)*hh.agrid[ia] + hh.w*hh.egrid[ie]
                    if cc<=0: cc = 1e-5
                    V_temp[ia,ie]   = hh.util(cc)   # VALUE FUNCTION
                    a1_temp[ia,ie]  = 0.0           # SAVING
        
        # GRID SEARCH AT AGE < T
        else:
            if comm.rank==0:
                V1 = hh.V[age+1,:,:]
            else:
                V1 = np.empty((hh.na,hh.ne),dtype=np.float64)                
            comm.Bcast(V1,root=0)

            # Split the for loop by workers
            lb  = int((comm.rank+0)*np.ceil((hh.na*hh.ne)/comm.size))
            ub  = int((comm.rank+1)*np.ceil((hh.na*hh.ne)/comm.size))
            if hh.na*hh.ne < ub:
                ub = hh.na*hh.ne
            leng = ub - lb
            Vp  = np.empty((int(leng),2))
            it  = 0

            for ind in range(lb,ub):
                Vp[it,:] = vfi_opt(hh, age, ind)
                it += 1

            # Gather the computed value function by each worker            
            comm.Gather(Vp,results,root=0)

            for ind in range(hh.na*hh.ne):
                ia = ind // hh.ne
                ie = ind % hh.ne
                V_temp[ia,ie]   = results[ind][0]   # VALUE FUNCTION
                a1_temp[ia,ie]  = results[ind][1]           # SAVING

        hh.set_V(age,V_temp)
        hh.set_a1(age,a1_temp)


        f2 = MPI.Wtime() - s2
        f0_sum  += f2
        if comm.rank==0:
            sys.stdout.write("Age: %d. Time: %f seconds. \n" % (age+1, round(f2, 4)))

        comm.Barrier()

        
    # TOTAL RUNTIME
    f0 = time.time() - s0
    run_time = [f0_sum,f0]
    
    return run_time


if __name__ == "__main__":
        
    comm = MPI.COMM_WORLD

    # Run time memory
    T        = 10    
    nrun     = 10
    
    # Iterate on the number of cores
    # Store the results as JSON data
    data = []
    for num_runs in range(nrun):
        
        if comm.rank==0:
            # Print out to a console window
            sys.stdout.write("####### %d times run with %d cores #######\n" % (num_runs+1, comm.size) )

        results = run_vfi(comm)
                
        # Store the total run time and initialization in a JSON data
        data.append({
                'core': int(comm.size),
                'run': int(num_runs),
                'age': int(0),
                'runtime': results[1]
        })
            
        comm.Barrier()
        
    
    if comm.rank==0:
        
        file_name = "runtime_oop_by_cores_mpi_" + str(comm.size) + ".txt"
        # Write the data to a output file
        with open(file_name,"w+") as outfile:
            json.dump(data, outfile)
    
    
        # Retrieve the results from the output file
        with open(file_name) as json_data:
            d = json.load(json_data)
            pprint(d)
        for ik in range(len(d)):
            sys.stdout.write("%d, %d, %d, %f \n" % (d[ik]['core'],d[ik]['run'],d[ik]['age'],d[ik]['runtime']))
