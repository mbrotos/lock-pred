#!/bin/bash

RESULTS_DIR="/Users/adamsorrenti/Documents/GitHub/lock-pred/results"
ZIPS_DIR="/Users/adamsorrenti/Library/CloudStorage/GoogleDrive-adam.sorrenti@torontomu.ca/My Drive/School/Research/lock_pred/Lock Prediction Results/sorted/"

ZIPS_only=("exp-21-lstm-sorted-table-locks.zip")

mkdir -p "$RESULTS_DIR"

for zip in "$ZIPS_DIR"/*.zip; do
    echo "Processing: $zip"

    # check if ZIPS_only is empty, if not, only process the zips in ZIPS_only
    if [ ${#ZIPS_only[@]} -gt 0 ]; then
        if [[ ! " ${ZIPS_only[@]} " =~ " ${zip##*/} " ]]; then
            echo "Skipping $zip"
            continue
        fi
    fi


    temp_dir=$(mktemp -d)
    # Extract only the two specific files from any exp-* folder under results/
    unzip "$zip" "results/exp-*/predictions.parquet" "results/exp-*/in_lock_sequences.parquet" -d "$temp_dir"
    # The extracted structure is now: temp_dir/results/exp-*/...
    if [ -d "$temp_dir/results" ]; then
        # Move each exp-* folder from temp_dir/results/ to your RESULTS_DIR
        mv "$temp_dir/results/"* "$RESULTS_DIR"
    else
        echo "No matching files found in $zip"
    fi
    rm -rf "$temp_dir"
done

echo "Extraction complete. Files are in $RESULTS_DIR"