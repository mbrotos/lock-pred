#!/bin/bash

# NOTE: This experiment is used to test naive baseline for each table separately.

# SLURM configurations (will be ignored if not running on SLURM)
#SBATCH --job-name=exp-19-naive-sorted-table-locks
#SBATCH --account=def-miranska
#SBATCH --output=logs/exp-19-naive-sorted-table-locks_%A_%a.out
#SBATCH --error=logs/exp-11-table-lock-naives_%A_%a.err
#SBATCH --array=0-43 # NOTE: Make sure this is equal to the number of configs
#SBATCH --time=23:59:00
#SBATCH --mem=64G
#SBATCH --mail-user=adam.sorrenti@torontomu.ca
#SBATCH --mail-type=ALL

# Activate virtual environment
source .venv/bin/activate

ITERATIONS=1

# Define experiment configurations
declare -a configs_base=(
    # Base configurations
    "--experiment_name exp-19-naive-sorted-table-locks/char_ --sort_by start_time --naive_baseline"
)

declare -a horizons=(1 2 3 4)

declare -a tables=("WAREHOUSE" "DISTRICT" "CUSTOMER" "ITEM" "ORDER_LINE" "STOCK" "ORDERS" "NEW_ORDER" "HISTORY") # "POLICY" "HMON_ATM_INFO")

# Generate configurations for each training data percentage
declare -a configs
for horizon in "${horizons[@]}"; do
    for table in "${tables[@]}"; do
        for config in "${configs_base[@]}"; do
            configs+=("$config --horizon $horizon --val_split 0.0 --model transformer --remove_system_tables --token_length_seq --data data/fixed/table_sep/$table.csv --naive_baseline")
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

# # Main loop
# for ((i=task_id; i<task_id+num_tasks; i++)); do
#     current_config=${configs[$i]}
#     echo "Running experiment: $current_config"

#     # Run the experiment for the specified number of iterations
#     for j in $(seq 1 $ITERATIONS); do
#         echo "Running iteration $j"
#         python src/train.py $current_config
#     done
# done

commands=()
for conf in "${configs[@]}"; do
    for i in $(seq 1 $ITERATIONS); do
        commands+=("python src/train.py $conf")
    done
done

# Finally, run all commands in parallel with 4 parallel jobs (edit -j 4 as you like)
printf '%s\n' "${commands[@]}" | parallel -j 4