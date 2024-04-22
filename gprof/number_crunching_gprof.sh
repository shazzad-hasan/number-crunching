#!/bin/bash

#SBATCH -N 1
#SBATCH --job-name="number_crunching_gprof"
#SBATCH -o number_crunching_gprof.%J.out
#SBATCH -e number_crunching_gprof.%J.err
#SBATCH -t 03:00:00
#SBATCH -p multi

g++ -march=native -fno-inline -O3 -pg -o number_crunching_gprof number_crunching.cpp

for k in {1..10}; do
    N=$((k * 10000))
    ./number_crunching_gprof $N  
    if [ "$k" -eq 10 ]; then
        gprof number_crunching_gprof gmon.out > profile_${k}.txt
        cp profile_${k}.txt number_crunching_gprof.out
    else
        gprof number_crunching_gprof gmon.out > profile_${k}.txt
    fi
done

echo "Profiling completed. The results for N = 100000 are in profile_10.txt and number_crunching_gprof.out."
