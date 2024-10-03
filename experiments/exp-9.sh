#!/bin/bash

# NOTE: This experiment is used to test the effect of remov

# SLURM configurations (will be ignored if not running on SLURM)
#SBATCH --job-name=exp-9
#SBATCH --account=def-miranska
#SBATCH --output=logs/exp-9_%A_%a.out
#SBATCH --error=logs/exp-9_%A_%a.err
#SBATCH --array=0-1 # NOTE: Make sure this is equal to the number of configs
#SBATCH --time=2:59:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --mail-user=adam.sorrenti@torontomu.ca
#SBATCH --mail-type=ALL

# Activate virtual environment
source .venv/bin/activate

ITERATIONS=10

# Define experiment configurations
declare -a configs=(
    "--experiment_name exp-9/small_nodupes --model transformer --remove_system_tables --token_length_seq --val_split 0.0 --data data/row_locks_nodupes.csv"
    "--experiment_name exp-9/large_nodupes --model transformer --remove_system_tables --token_length_seq --val_split 0.0 --data data/row_locks_large_nodupes.csv"
)

# Determine if running on SLURM or locally
if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    # Running on SLURM
    task_id=$SLURM_ARRAY_TASK_ID
    num_tasks=1
else
    # Running locally
    task_id=0
    num_tasks=${#configs[@]}
fi

# Main loop
for ((i=task_id; i<task_id+num_tasks; i++)); do
    current_config=${configs[$i]}
    echo "Running experiment: $current_config"

    # Run the experiment for the specified number of iterations
    for j in $(seq 1 $ITERATIONS); do
        echo "Running iteration $j"
        python src/train.py $current_config
    done
done