#!/bin/bash

# NOTE: This experiements test causal transformer model at increasing horizons

# SLURM configurations (will be ignored if not running on SLURM)
#SBATCH --job-name=exp-14
#SBATCH --account=def-miranska
#SBATCH --output=logs/exp-14_%A_%a.out
#SBATCH --error=logs/exp-14_%A_%a.err
#SBATCH --gres=gpu:a100_3g.20gb:1 
#SBATCH --time=11:59:00
#SBATCH --mem=32G
#SBATCH --array=0-3
#SBATCH --cpus-per-task=1
#SBATCH --mail-user=adam.sorrenti@torontomu.ca
#SBATCH --mail-type=ALL

# Activate virtual environment
source .venv/bin/activate

horizons=(1 2 3 4)

cur_horizon=${horizons[$SLURM_ARRAY_TASK_ID]}
echo "Current horizon: $cur_horizon"

python src/train.py --experiment_name exp-14/ --model transformer_causal --remove_system_tables --token_length_seq --val_split 0.1 --data data/fixed/row_locks.csv --horizon $cur_horizon --seq_length 128 

