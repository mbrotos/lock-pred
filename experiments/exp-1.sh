#!/bin/bash

rm -rf results/exp-1

ITERATIONS=10

for i in $(seq 1 $ITERATIONS); do
    echo "Running iteration $i"

    # char models
    ## Transformer
    python src/train.py --experiment_name "exp-1/char_"
    python src/train.py --experiment_name "exp-1/char_se_" --add_start_end_tokens
    python src/train.py --experiment_name "exp-1/char_r_" --add_row_id
    python src/train.py --experiment_name "exp-1/char_lr_" --add_label_tokens
    python src/train.py --experiment_name "exp-1/char_se_r_" --add_start_end_tokens --add_row_id
    python src/train.py --experiment_name "exp-1/char_se_lr_" --add_start_end_tokens --add_label_tokens
    python src/train.py --experiment_name "exp-1/char_r_lr_" --add_row_id --add_label_tokens
    python src/train.py --experiment_name "exp-1/char_se_r_lr_" --add_start_end_tokens --add_row_id --add_label_tokens

    ## LSTM
    python src/train.py --experiment_name "exp-1/char_" --model "lstm"
    python src/train.py --experiment_name "exp-1/char_se_" --model "lstm" --add_start_end_tokens
    python src/train.py --experiment_name "exp-1/char_r_" --model "lstm" --add_row_id
    python src/train.py --experiment_name "exp-1/char_lr_" --model "lstm" --add_label_tokens
    python src/train.py --experiment_name "exp-1/char_se_r_" --model "lstm" --add_start_end_tokens --add_row_id
    python src/train.py --experiment_name "exp-1/char_se_lr_" --model "lstm" --add_start_end_tokens --add_label_tokens
    python src/train.py --experiment_name "exp-1/char_r_lr_" --model "lstm" --add_row_id --add_label_tokens
    python src/train.py --experiment_name "exp-1/char_se_r_lr_" --model "lstm" --add_start_end_tokens --add_row_id --add_label_tokens

    # word models
    ## Transformer
    python src/train.py --experiment_name "exp-1/word_" --tokenization "word"
    python src/train.py --experiment_name "exp-1/word_se_" --tokenization "word" --add_start_end_tokens
    python src/train.py --experiment_name "exp-1/word_r_" --tokenization "word" --add_row_id
    python src/train.py --experiment_name "exp-1/word_lr_" --tokenization "word" --add_label_tokens
    python src/train.py --experiment_name "exp-1/word_se_r_" --tokenization "word" --add_start_end_tokens --add_row_id
    python src/train.py --experiment_name "exp-1/word_se_lr_" --tokenization "word" --add_start_end_tokens --add_label_tokens
    python src/train.py --experiment_name "exp-1/word_r_lr_" --tokenization "word" --add_row_id --add_label_tokens
    python src/train.py --experiment_name "exp-1/word_se_r_lr_" --tokenization "word" --add_start_end_tokens --add_row_id --add_label_tokens

    ## LSTM
    python src/train.py --experiment_name "exp-1/word_" --tokenization "word" --model "lstm"    
    python src/train.py --experiment_name "exp-1/word_se_" --tokenization "word" --model "lstm" --add_start_end_tokens
    python src/train.py --experiment_name "exp-1/word_r_" --tokenization "word" --model "lstm" --add_row_id
    python src/train.py --experiment_name "exp-1/word_lr_" --tokenization "word" --model "lstm" --add_label_tokens
    python src/train.py --experiment_name "exp-1/word_se_r_" --tokenization "word" --model "lstm" --add_start_end_tokens --add_row_id
    python src/train.py --experiment_name "exp-1/word_se_lr_" --tokenization "word" --model "lstm" --add_start_end_tokens --add_label_tokens
    python src/train.py --experiment_name "exp-1/word_r_lr_" --tokenization "word" --model "lstm" --add_row_id --add_label_tokens
    python src/train.py --experiment_name "exp-1/word_se_r_lr_" --tokenization "word" --model "lstm" --add_start_end_tokens --add_row_id --add_label_tokens
    
done
