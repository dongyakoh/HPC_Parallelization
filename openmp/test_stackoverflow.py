# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 21:36:56 2018

@author: Don Koh
"""

import joblib
from multiprocessing import Pool
from joblib import Parallel, delayed
import time
import numpy as np
import sys
 
print(joblib.__version__)


def doubler(number):
    time.sleep(0.01)
    return number * 2


def main(idx,num_proc):
    
    num_elements = 1000
    num_iter     = 2

    # multiprocessing    
    if idx == 0:
        x = np.arange(1,num_elements,1,int)
        print("Module: multiprocessing.Pool")
        chunk = int(num_elements/(4*num_proc))
        pool = Pool(processes=num_proc)
        for iter in range(num_iter):
            t1 = time.time()
            result = pool.map(doubler, x, chunk)
            print("num_proc =", num_proc, ", runtime = ", time.time() - t1)
        pool.close()    
        
    # joblib
    elif idx == 1:
        print("Module: joblib.Parallel")
        chunk = int(num_elements/(4*num_proc))
#        with Parallel(n_jobs=num_proc,backend='multiprocessing',batch_size='auto',prefer='processes',max_nbytes=None,verbose=10) as parallel:
        with Parallel(n_jobs=num_proc,backend='multiprocessing',batch_size=chunk,pre_dispatch=chunk,prefer='processes',max_nbytes=None,verbose=20) as parallel:
            for iter in range(num_iter):    
                t1 = time.time()
                result = parallel(delayed(doubler)(x) for x in np.arange(1,num_elements,1,int))        
                print("num_proc =", num_proc, ", runtime = ", time.time() - t1)
        
        
if __name__ == '__main__':
    
    
    idx = 0
    num_proc = 2
    if len(sys.argv) > 1:
        idx = int(sys.argv[1])
        num_proc = int(sys.argv[2])
    main(idx,num_proc)

    
