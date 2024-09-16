#!/bin/bash

# NOTE: This experiment is used to test the effect of different data preparation configurations and training data percentages.
# It includes variations with and without start/end tokens, row IDs, and label tokens.
# The experiment also tests different percentages of training data (100%, 80%, 60%, 40%, 20%) to analyze the impact on model performance.
# All configurations are run without a validation split to maximize training data usage.
# All configurations are run with 10 iterations to get a good average of the performance.
# All configurations are run with the char-based tokenization.
# All configurations are run with the transformer model.
# All configurations are run with the remove_system_tables flag set to true.

# SLURM configurations (will be ignored if not running on SLURM)
#SBATCH --job-name=exp-5
#SBATCH --account=def-miranska
#SBATCH --output=logs/exp-5_%A_%a.out
#SBATCH --error=logs/exp-5_%A_%a.err
#SBATCH --array=0-39 # NOTE: Make sure this is equal to the number of configs
#SBATCH --time=11:59:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --mail-user=adam.sorrenti@torontomu.ca
#SBATCH --mail-type=ALL

# Activate virtual environment
source .venv/bin/activate

ITERATIONS=10

# Define experiment configurations
declare -a configs_base=(
    # Base configurations
    "--experiment_name exp-5/char_"
    "--experiment_name exp-5/char_se_ --add_start_end_tokens "
    "--experiment_name exp-5/char_r_ --add_row_id "
    "--experiment_name exp-5/char_lr_ --add_label_tokens "
    "--experiment_name exp-5/char_se_r_ --add_start_end_tokens --add_row_id "
    "--experiment_name exp-5/char_se_lr_ --add_start_end_tokens --add_label_tokens "
    "--experiment_name exp-5/char_r_lr_ --add_row_id --add_label_tokens "
    "--experiment_name exp-5/char_se_r_lr_ --add_start_end_tokens --add_row_id --add_label_tokens "
)

# Define the training data percentages
declare -a train_data_percents=(1.0 0.8 0.6 0.4 0.2)

# Generate configurations for each training data percentage
declare -a configs
for percent in "${train_data_percents[@]}"; do
    for config in "${configs_base[@]}"; do
        configs+=("$config --train_data_percent_used $percent --val_split 0.0 --model transformer --remove_system_tables --token_length_seq")
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