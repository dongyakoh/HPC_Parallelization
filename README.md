# HPC_Parallelization
Python codes for "HPC-Efficient Parallel Computing in Economics" by Dongya Koh.

This directory contains the following contents:

1. openmp - a folder that contains programs that compute parallel computing times with OpenMP (Section 3.2) and by passing objects vs. variables (Section 4.3).
		vfi_by_cores_oop.py - runs a program with OpenMP that passes objects to each worker.
		vfi_by_cores_pop.py - runs a program with OpenMP that passes variables to each worker.
		plot_vfi_cores.py - generates Figure 2 in the main draft.
		plot_vfi_cores_oop_pop.py - generates Figure 8 in the main draft.
		
2. mpi - a folder that contains programs that executes parallel computing with MPI (Section 3.3)
		vfi_oop_by_cores_mpi.py - runs a program with MPI that passes objects to each worker.
		plot_oop_by_cores_mpi.py - plots Figure 3 in the main draft.
		
3. partition - a folder that contains parallel computing programs that partition a discrete state space into pieces (Section 4.1)
		vfi_oop_partition_by_cores.py - runs an object-oriented program in parallel with OpenMP which partitions a state space into the number of workers.
		vfi_oop_nonpartition_by_cores.py - runs an object-oriented program in parallel with OpenMP without partitioning a state space for a comparison to the partitioned program.
		plot_vfi_oop_partition_by_cores.py - plots Figure 5.
		plot_error_oop_partition_by_cores.py - plots Figure 6.
		
4. chunk - a folder that contains programs that assigns tasks with different chunk sizes (Section 4.2)
		vfi_by_cores_chunks_oop.py - runs an object-oriented program in parallel with OpenMP that assign a different chunk size of tasks to multiple workers.
		plot_vfi_cores_chunks_by_core.py - plots Figure 7.
		
Created by Dongya Koh 9/10/2019
