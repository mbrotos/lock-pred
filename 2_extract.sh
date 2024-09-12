#! /bin/bash

source .venv/bin/activate

python src/extract.py \
    --experiment_name exp-test \
    --output_file "results.csv"