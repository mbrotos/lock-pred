#!/bin/bash

# NOTE: This experiment is used to test training a model for each table separately.

# SLURM configurations (will be ignored if not running on SLURM)
#SBATCH --job-name=exp-37-lstm-local-rounded
#SBATCH --account=def-miranska
#SBATCH --output=logs/exp-37-lstm-local-rounded%A_%a.out
#SBATCH --error=logs/exp-37-lstm-local-rounded%A_%a.err
#SBATCH --array=0-31 # NOTE: Make sure this is equal to the number of configs
#SBATCH --time=23:59:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=1
#SBATCH --mail-user=adam.sorrenti@torontomu.ca
#SBATCH --mail-type=ALL

# Activate virtual environment
module load gcc arrow
module load python/3.11
source ~/.venv/bin/activate

ITERATIONS=10

# Define experiment configurations
declare -a configs_base=(
    # Base configurations
    "--experiment_name exp-37-lstm-local-rounded/char_ --sort_by start_time-dedupe --rounding_bin_size 10000"
)

declare -a horizons=(1 2 3 4)

declare -a tables=("DISTRICT" "STOCK" "ORDER_LINE" "ORDERS" "NEW_ORDER" "WAREHOUSE" "CUSTOMER" "HISTORY")

# Generate configurations for each training data percentage
declare -a configs
for horizon in "${horizons[@]}"; do
    for table in "${tables[@]}"; do
        for config in "${configs_base[@]}"; do
            configs+=("$config --horizon $horizon --val_split 0.2 --checkpoint --model lstm --remove_system_tables --token_length_seq --data data/fixed/row_sep/$table.csv")
        done
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