#!/bin/bash

# NOTE: This experiment runs LSTM models on the row locks and table locks datasets with different horizons.

# SLURM configurations (will be ignored if not running on SLURM)
#SBATCH --job-name=add_time_array
#SBATCH --account=def-miranska
#SBATCH --output=logs/add_time_%A_%a.out
#SBATCH --error=logs/add_time_%A_%a.err
#SBATCH --time=11:59:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=1
#SBATCH --mail-user=adam.sorrenti@torontomu.ca
#SBATCH --mail-type=ALL
#SBATCH --array=0-773  # Replace 4 with (number of args.json files - 1)

# Base directory
BASE_DIR="results/results_old/fixed"

# Exclude directories
EXCLUDE_DIRS=("exp-6-row_id" "exp-6-table-locks-longer" "table_sep")

# Build the find command with exclusions
FIND_CMD="find \"$BASE_DIR\" -type f -name \"args.json\""
for EXCLUDE_DIR in "${EXCLUDE_DIRS[@]}"; do
    FIND_CMD+=" ! -path \"*/$EXCLUDE_DIR/*\""
done

# Find all args.json files and store them in an array
args_files=()
while IFS=  read -r -d $'\0' file; do
    args_files+=("$file")
done < <(eval "$FIND_CMD" -print0)

# Get the total number of files found
num_files=${#args_files[@]}
echo "Found $num_files args.json files."

# Check if SLURM_ARRAY_TASK_ID is set (running as part of an array job)
if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    # Get the index from SLURM_ARRAY_TASK_ID
    index=$SLURM_ARRAY_TASK_ID

    # Check if index is within bounds
    if [[ $index -ge 0 && $index -lt $num_files ]]; then
        args_file="${args_files[$index]}"
        args_dir=$(dirname "$args_file")
        new_args_file="$args_dir/args_save_timing.json"
        jq --arg save_times_exit "$args_dir" '. + {save_times_exit: $save_times_exit}' "$args_file" > "$new_args_file"
        echo "Created $new_args_file (Job $SLURM_ARRAY_TASK_ID)"
        python src/train.py --args_file "$new_args_file"
    else
        echo "Error: SLURM_ARRAY_TASK_ID ($SLURM_ARRAY_TASK_ID) is out of bounds."
    fi
else
    echo "Error: This script is intended to be run as a SLURM array job."
    echo "The SLURM_ARRAY_TASK_ID environment variable is not set."
    exit 1
fi