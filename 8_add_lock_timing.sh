#!/bin/bash

# Base directory
BASE_DIR="results_old/fixed"

# Exclude directories
EXCLUDE_DIRS=("exp-10" "exp-6-row_id" "exp-6-table-locks-longer" "table_sep")

# Build the find command with exclusions
FIND_CMD="find \"$BASE_DIR\" -type f -name \"args.json\""
for EXCLUDE_DIR in "${EXCLUDE_DIRS[@]}"; do
    FIND_CMD+=" ! -path \"*/$EXCLUDE_DIR/*\""
done

# Function to process each args.json file
process_file() {
    args_file="$1"
    args_dir=$(dirname "$args_file")
    new_args_file="$args_dir/args_save_timing.json"
    jq --arg save_times_exit "$args_dir" '. + {save_times_exit: $save_times_exit}' "$args_file" > "$new_args_file"
    echo "Created $new_args_file"
    python src/train.py --args_file "$new_args_file"
}

export -f process_file

# Execute the find command and process the results in parallel
# use -j to limit the number of parallel processes (-j 1 for sequential execution)
eval $FIND_CMD | parallel process_file