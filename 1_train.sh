#! /bin/bash

source .venv/bin/activate


# This command includes all the default values for the arguments defined in the parse_args function.
# Note that boolean flags (like --shuffle, --add_start_end_tokens, etc.) are not included
# because their default value is False, and they don't need to be specified unless you want to enable them.
python src/train.py \
    --experiment_name exp-test/ \
    --model transformer \
    --data data/row_locks.csv \
    --train_data_percent_used 1.0 \
    --epochs 30 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --seq_length 50 \
    --test_split 0.3 \
    --val_split 0.3 \
    --vocab_size 900 \
    --tokenization char \
    --patience 5 \
    --results_dir results

experiments=("exp-1" "exp-2" "exp-3" "exp-4" "exp-5" "exp-5-large")
for experiment in "${experiments[@]}"; do
    sbatch experiments/$experiment.sh
    # OR, for local testing
    # bash experiments/$experiment.sh
done