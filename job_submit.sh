#!/bin/bash
#SBATCH --partition=exfel
#SBATCH --time=0-20:00:00                           # Maximum time requested
#SBATCH --nodes=1  #5
#SBATCH --constraint="[Gold-6140|Gold-6240]&768G"                               # Number of nodes
#SBATCH --chdir /gpfs/exfel/theory_group/user/dasarina/DATA_optical/DATA_DOPED/DATA_TEST/DATA_22   # directory must already exist!
#SBATCH --job-name  D-TRILEX
#SBATCH --output  hostname-%N-%j.out            # File to which STDOUT will be written
#SBATCH --error  hostname-%N-%j.err            # File to which STDERR will be written 
export OMP_NUM_THREADS=1
#for i in $(seq 177 177)
#do
#time /usr/lib64/openmpi/bin/mpirun -mca pml ucx -mca btl '^uct,ofi,openib' -mca mtl '^ofi' -np 66 /gpfs/exfel/theory_group/user/dasarina/ppsc/cbuild/programs/DMFT_2d_square.ex ./param.in ./
time /usr/lib64/openmpi/bin/mpirun -mca pml ucx -mca btl '^uct,ofi,openib' -mca mtl '^ofi' -np 66 /gpfs/exfel/theory_group/user/dasarina/ppsc/cbuild/programs/DGW.ex ./param.in ./
#var3=$(echo $i\* 1 + 1 | bc)
#cd ../DATA_$var3
#done

