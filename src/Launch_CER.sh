#!/bin/bash
#SBATCH --partition=partition
#SBATCH --time=3-00:00:00
#SBATCH --mem=0
#SBATCH -N 1
#SBATCH -o ./job_outputs/output.out
#SBATCH -e ./job_outputs/error.err

. /env/activate myenv
python3 CER_Benchmark.sh $1 $2
