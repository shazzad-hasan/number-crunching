#!/bin/bash

#SBATCH -N 1
#SBATCH --job-name="number_crunching_likwid"
#SBATCH -o number_crunching_likwid.%J.out
#SBATCH -e number_crunching_likwid.%J.err
#SBATCH -t 03:00:00
#SBATCH -p multi

source /etc/profile.d/modules.sh

module load likwid/5.2.0

g++ -std=c99 -mfma -O1 -DLIKWID_PERFMON -fno-inline -march=native -o number_crunching_likwid number_crunching_likwid.cpp -llikwid

likwid-perfctr -m -g MEM_DP -C 0 ./number_crunching_likwid 
likwid-perfctr -m -g FLOPS_DP -C 0 ./number_crunching_likwid 

