#! /bin/bash

source .venv/bin/activate

# Transform the data is required for experiments with multiple iterations
python src/transform.py \
    --input_file "results/exp-test/results.csv" \
    --output_file "results_clean.csv"

experiments=("exp-1" "exp-2" "exp-3" "exp-4" "exp-5" "exp-5-large")
for experiment in "${experiments[@]}"; do
    python src/transform.py \
        --input_file "results/$experiment/results.csv" \
        --output_file "results_clean.csv"
done
