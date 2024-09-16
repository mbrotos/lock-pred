#! /bin/bash

source .venv/bin/activate

python src/extract.py \
    --experiment_name exp-test \
    --output_file "results.csv"

experiments=("exp-1" "exp-2" "exp-3" "exp-4" "exp-5" "exp-5-large")
for experiment in "${experiments[@]}"; do
    python src/extract.py \
        --experiment_name $experiment \
        --output_file "results.csv"
done