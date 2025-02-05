#!/bin/bash

# Base directory
BASE_DIR="results/fixed"

# Find all args.json files excluding those in exp-10 folder
find "$BASE_DIR" -type f -name "args.json" ! -path "*/exp-10/*" | while read -r args_file; do
    # Determine the directory containing the args.json file
    args_dir=$(dirname "$args_file")
    
    # Path to the model.keras file
    model_weights_path="$args_dir/model.keras"
    
    # Check if model.keras file exists
    if [ -f "$model_weights_path" ]; then
        # Create a new file with the model_weights field added
        new_args_file="$args_dir/args_with_weights.json"
        jq --arg model_weights "$model_weights_path" '. + {model_weights: $model_weights}' "$args_file" > "$new_args_file"
        echo "Created $new_args_file with model_weights: $model_weights_path"
    else
        echo "model.keras not found in $args_dir"
    fi
done