#!/bin/bash

#SBATCH --time=00:10:00
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=24
#SBATCH --gres=gpu:4
#SBATCH --job-name=instructor
#SBATCH --output=cme213.out
#SBATCH --error=cme213.err

### ---------------------------------------
### BEGINNING OF EXECUTION
### ---------------------------------------
echo The master node of this job is `hostname`
echo This job runs on the following nodes:
echo `scontrol show hostname $SLURM_JOB_NODELIST`
echo "Starting at `date`"
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running on $SLURM_NPROCS processors."
echo "Current working directory is `echo $SLURM_SUBMIT_DIR`"
echo
echo Output from code
echo ----------------
### end of information preamble

cd $SLURM_SUBMIT_DIR

EXE="./mpi_prime"
mpiexec -n 1 $EXE
mpiexec -n 2 $EXE
mpiexec -n 4 $EXE
mpiexec -n 8 $EXE
mpiexec -n 16 $EXE
mpiexec -n 24 $EXE
mpiexec -n 48 $EXE
mpiexec -n 96 $EXE
