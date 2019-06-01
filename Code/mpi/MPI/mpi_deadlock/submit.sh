#!/bin/bash

#PBS -N cme213 
#PBS -q gpu
#PBS -j oe
#PBS -o cme213.out 
#PBS -l nodes=1:ppn=24

EXE=./mpi_hello

# Number of processes
NPROCS=4
# Number of processes per node
PPN=4


echo "Running using hosts:"
cat $PBS_NODEFILE | uniq

cd $PBS_O_WORKDIR

echo -------------------------------
echo "mpiexec -bind-to hwthread -map-by socket"
HYDRA_TOPO_DEBUG=1 mpiexec -bind-to hwthread -map-by socket -np $NPROCS -ppn $PPN $EXE

echo -------------------------------
echo "mpiexec -bind-to hwthread -map-by numa"
HYDRA_TOPO_DEBUG=1 mpiexec -bind-to hwthread -map-by numa -np $NPROCS -ppn $PPN $EXE

echo -------------------------------
echo "mpiexec -bind-to hwthread -map-by core"
HYDRA_TOPO_DEBUG=1 mpiexec -bind-to hwthread -map-by core -np $NPROCS -ppn $PPN $EXE

echo -------------------------------
echo "mpiexec -bind-to hwthread -map-by hwthread"
HYDRA_TOPO_DEBUG=1 mpiexec -bind-to hwthread -map-by hwthread -np $NPROCS -ppn $PPN $EXE

echo -------------------------------
echo "mpiexec -bind-to socket -map-by socket"
HYDRA_TOPO_DEBUG=1 mpiexec -bind-to socket -map-by socket -np $NPROCS -ppn $PPN $EXE

echo -------------------------------
echo "mpiexec -bind-to socket -map-by core"
HYDRA_TOPO_DEBUG=1 mpiexec -bind-to socket -map-by core -np $NPROCS -ppn $PPN $EXE
