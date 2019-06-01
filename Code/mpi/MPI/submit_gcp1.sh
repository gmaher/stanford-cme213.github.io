#!/bin/bash

echo The master node of this job is `hostname`
echo "Starting at `date`"
echo Output from code
echo ----------------
### end of information preamble

EXE="./mpi_hello"

NPROCS="-mca btl ^openib -n 6" 

echo ----------------

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
