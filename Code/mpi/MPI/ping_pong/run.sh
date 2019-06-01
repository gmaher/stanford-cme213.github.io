#!/bin/bash
for ((n=1;n<8;n++))
do
    mpirun -mca btl ^openib -n 8 ./ping_pong -p $n
done    