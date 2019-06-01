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

EXE="./mpi_hello"

MPIEXEC="mpiexec"

NPROCS="-n 4"
PPN="--npernode 4"
CMD="$MPIEXEC $NPROCS $PPN $EXE"
echo $CMD
$CMD

PPN="--npernode 1"
CMD="$MPIEXEC $NPROCS $PPN $EXE"
echo $CMD
$CMD

echo ----------------

# Number of processes
NPROCS="-n 6"

# mpi exec command
MPIEXEC="mpiexec --report-bindings --oversubscribe"

# bind to options
# none, hwthread, core, l1cache, l2cache, l3cache, socket, numa, board

# map=by
# slot, hwthread, core, socket, numa, board, node

echo -------------------------------
CMD="$MPIEXEC $NPROCS --bind-to hwthread --map-by socket $EXE"
echo $CMD
$CMD

echo -------------------------------
CMD="$MPIEXEC $NPROCS --bind-to hwthread --map-by core $EXE"
echo $CMD
$CMD

echo -------------------------------
CMD="$MPIEXEC $NPROCS --bind-to hwthread --map-by hwthread $EXE"
echo $CMD
$CMD

echo -------------------------------
CMD="$MPIEXEC $NPROCS --bind-to socket --map-by socket $EXE"
echo $CMD
$CMD

echo -------------------------------
CMD="$MPIEXEC $NPROCS --bind-to socket --map-by core $EXE"
echo $CMD
$CMD
