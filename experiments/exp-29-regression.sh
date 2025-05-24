#!/bin/bash

# SLURM configurations (will be ignored if not running on SLURM)
#SBATCH --job-name=exp-29-regression
#SBATCH --account=def-miranska
#SBATCH --output=logs/exp-29-regression%A_%a.out
#SBATCH --error=logs/exp-29-regression%A_%a.err
#SBATCH --array=0-0 # NOTE: Make sure this is equal to the number of configs
#SBATCH --time=11:59:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --cpus-per-task=1
#SBATCH --mail-user=adam.sorrenti@torontomu.ca
#SBATCH --mail-type=ALL

# Activate virtual environment
module load gcc arrow
module load python/3.11
source ~/.venv/bin/activate

cd src/

jupyter nbconvert --to notebook --execute regression.ipynb --output regression-done3.ipynb
