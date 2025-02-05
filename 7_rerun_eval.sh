#!/bin/bash

# Base directory
BASE_DIR="results/fixed"

# Find all args.json files excluding those in exp-10 folder
find "$BASE_DIR" -type f -name "args.json" ! -path "*/exp-10/*" | while read -r args_file; do
    # Determine the directory containing the args.json file
    args_dir=$(dirname "$args_file")
    
    # Path to the model.keras file
    args_weights="$args_dir/args_with_weights.json"
    
    # Check if model.keras file exists
    if [ -f "$args_weights" ]; then
        echo "Running w/ $args_weights"
        python src/train.py --args_file "$args_weights"
    else
        echo "No args_with_weights.json found in $args_dir"
    fi
done