#!/bin/bash

# NOTE: This experiment is used to test all the sequence creation methods with an increased sequence length.

# SLURM configurations (will be ignored if not running on SLURM)
#SBATCH --job-name=exp-7
#SBATCH --account=def-miranska
#SBATCH --output=logs/exp-7_%A_%a.out
#SBATCH --error=logs/exp-7_%A_%a.err
#SBATCH --array=0 # NOTE: Make sure this is equal to the number of configs
#SBATCH --time=23:59:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --mail-user=adam.sorrenti@torontomu.ca
#SBATCH --mail-type=ALL

# Activate virtual environment
source .venv/bin/activate

ITERATIONS=10

# Define experiment configurations
declare -a configs=(
    "--experiment_name exp-7/TTT60_ --val_split 0.0 --model transformer --remove_system_tables \
    --data data/row_locks_large.csv --token_length_seq --seq_length 60 \
    --add_start_end_tokens --add_row_id --add_label_tokens"
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