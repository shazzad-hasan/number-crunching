#!/bin/bash

#SBATCH -N 1
#SBATCH --job-name="number_crunching_gprof"
#SBATCH -o number_crunching_gprof.%J.out
#SBATCH -e number_crunching_gprof.%J.err
#SBATCH -t 03:00:00
#SBATCH -p multi

# Compile the program with profiling enabled
g++ -march=native -fno-inline -O3 -pg -o number_crunching_gprof number_crunching.cpp

# Loop to run the program with different N values and profile it
for k in {1..10}; do
    N=$((k * 10000))
    ./number_crunching_gprof $N
    gprof number_crunching_gprof gmon.out > profile_${k}.txt
done