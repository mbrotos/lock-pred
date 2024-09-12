#! /bin/bash

source .venv/bin/activate

# Transform the data is required for experiments with multiple iterations
python src/transform.py \
    --input_file "results/exp-test/results.csv" \
    --output_file "results_clean.csv"