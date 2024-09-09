#! /bin/bash

source .venv/bin/activate
# Char-based tokenization
python src/train.py \
    --model transformer \
    --tokenization char

python src/train.py \
    --model lstm \
    --tokenization char

# Word-based tokenization
python src/train.py \
    --model transformer \
    --tokenization word \
    --vocab_size 1000 \
    --out_seq_length 2

python src/train.py \
    --model lstm \
    --tokenization word \
    --vocab_size 1000 \
    --out_seq_length 2
    
