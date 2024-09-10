#!/bin/bash

# SLURM configurations (will be ignored if not running on SLURM)
#SBATCH --job-name=exp-1
#SBATCH --account=def-miranska
#SBATCH --output=logs/exp-1_%A_%a.out
#SBATCH --error=logs/exp-1_%A_%a.err
#SBATCH --array=0-31
#SBATCH --time=2:59:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mail-user=adam.sorrenti@torontomu.ca
#SBATCH --mail-type=ALL

# Activate virtual environment
source .venv/bin/activate

ITERATIONS=10

# Define experiment configurations
declare -a configs=(
    # shuffled
    # char tokenization
    "--experiment_name exp-2/shuffled_char_ --model transformer --shuffle" 
    "--experiment_name exp-2/shuffled_char_ --model lstm --shuffle"

    # word tokenization
    "--experiment_name exp-2/shuffled_word_ --model transformer --tokenization word --shuffle" 
    "--experiment_name exp-2/shuffled_word_ --model lstm --tokenization word --shuffle"

    # unshuffled
    # char tokenization
    "--experiment_name exp-2/unshuffled_char_ --model transformer" 
    "--experiment_name exp-2/unshuffled_char_ --model lstm"

    # word tokenization
    "--experiment_name exp-2/unshuffled_word_ --model transformer --tokenization word" 
    "--experiment_name exp-2/unshuffled_word_ --model lstm --tokenization word"

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