#!/bin/bash

# NOTE: This experiment is used to test the effect modeling using a naive baseline.

# SLURM configurations (will be ignored if not running on SLURM)
#SBATCH --job-name=exp-10
#SBATCH --account=def-miranska
#SBATCH --output=logs/exp-10_%A_%a.out
#SBATCH --error=logs/exp-10_%A_%a.err
#SBATCH --array=0-15 # NOTE: Make sure this is equal to the number of configs
#SBATCH --time=2:59:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --mail-user=adam.sorrenti@torontomu.ca
#SBATCH --mail-type=ALL

# Activate virtual environment
source .venv/bin/activate

ITERATIONS=1 # NOTE: The std should be 0 since the naive baseline is deterministic.

# Define experiment configurations
declare -a configs_base=(
    "--experiment_name exp-10/row_naive_small --model transformer --remove_system_tables --token_length_seq --val_split 0.0 --data data/row_locks.csv --naive_baseline"
    "--experiment_name exp-10/row_naive_large --model transformer --remove_system_tables --token_length_seq --val_split 0.0 --data data/row_locks_large.csv --naive_baseline"
    "--experiment_name exp-10/table_naive_small --model transformer --remove_system_tables --token_length_seq --val_split 0.0 --data data/table_locks.csv --naive_baseline"
    "--experiment_name exp-10/table_naive_large --model transformer --remove_system_tables --token_length_seq --val_split 0.0 --data data/table_locks_large.csv --naive_baseline"
)

# Define the training data percentages
declare -a horizons=(1 2 3 4)

# Generate configurations for each training data percentage
declare -a configs
for horizon in "${horizons[@]}"; do
    for config in "${configs_base[@]}"; do
        configs+=("$config --horizon $horizon")
    done
done

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