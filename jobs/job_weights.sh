#!/bin/bash
# SLURM submission script for multiple serial jobs on Niagara
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --time=24:00:00
#SBATCH --job-name tANS_weights
#SBATCH --output=/home/l/lungboy/tadam/scratch/Moshovos/ANS_NeuralComp/jobs/logs/tANS_weights_%j.out

module load NiaEnv/2019b python/3.11.5

source ~/.virtualenvs/pytorch_env/bin/activate

echo "Running tANS testing with weights"
python scripts/tANS_testing_weights_parallel.py

echo "Done"