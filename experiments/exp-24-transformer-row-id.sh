#!/bin/bash

# NOTE: This experiment runs LSTM models on the row locks and table locks datasets with different horizons.

# SLURM configurations (will be ignored if not running on SLURM)
#SBATCH --job-name=exp-24-transformer-row-id
#SBATCH --account=def-miranska
#SBATCH --output=logs/exp-24-transformer-row-id_%A_%a.out
#SBATCH --error=logs/exp-24-transformer-row-id_%A_%a.err
#SBATCH --array=0-7 # NOTE: Make sure this is equal to the number of configs
#SBATCH --time=23:59:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=1
#SBATCH --mail-user=adam.sorrenti@torontomu.ca
#SBATCH --mail-type=ALL

# Activate virtual environment
source .venv/bin/activate

ITERATIONS=10

# Define experiment configurations
declare -a configs_base=(
    # Base configurations
    "--experiment_name exp-24-transformer-row-locks-row-id/char_ --data data/fixed/row_locks.csv --sort_by start_time"
    "--experiment_name exp-24-transformer-table-locks-row-id/char_ --data data/fixed/table_locks.csv --sort_by start_time"
)

# Define the training data percentages
declare -a horizons=(1 2 3 4)

# Generate configurations for each training data percentage
declare -a configs
for horizon in "${horizons[@]}"; do
    for config in "${configs_base[@]}"; do
        configs+=("$config --horizon $horizon --val_split 0.0 --model transformer --remove_system_tables --token_length_seq --add_row_id --add_label_tokens")
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