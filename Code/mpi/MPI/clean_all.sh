#!/bin/bash

function cleanUp {
echo "Cleaning up in directory: `pwd`"
make clean
rm -f cme213.err
rm -f cme213.out
}

dirlist="groups matvec2D mmm mpi_cart mpi_nonblocking samplesort matvecrow mpi_all_reduce mpi_deadlock prime"

cleanUp

for d in $dirlist
do
  cd $d
  cleanUp
  cd ..
done
