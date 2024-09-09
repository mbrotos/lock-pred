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
    --tokenization word 

python src/train.py \
    --model lstm \
    --tokenization word 
    
