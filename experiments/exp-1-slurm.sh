#!/bin/bash

#SBATCH --job-name=exp-1
#SBATCH --output=logs/exp-1_%A_%a.out
#SBATCH --error=logs/exp-1_%A_%a.err
#SBATCH --array=0-31
#SBATCH --time=2:59:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1

source .venv/bin/activate

ITERATIONS=10

# Define experiment configurations
declare -a configs=(
    # Char models - Transformer
    "--experiment_name exp-1/char_"
    "--experiment_name exp-1/char_se_ --add_start_end_tokens"
    "--experiment_name exp-1/char_r_ --add_row_id"
    "--experiment_name exp-1/char_lr_ --add_label_tokens"
    "--experiment_name exp-1/char_se_r_ --add_start_end_tokens --add_row_id"
    "--experiment_name exp-1/char_se_lr_ --add_start_end_tokens --add_label_tokens"
    "--experiment_name exp-1/char_r_lr_ --add_row_id --add_label_tokens"
    "--experiment_name exp-1/char_se_r_lr_ --add_start_end_tokens --add_row_id --add_label_tokens"
    
    # Char models - LSTM
    "--experiment_name exp-1/char_ --model lstm"
    "--experiment_name exp-1/char_se_ --model lstm --add_start_end_tokens"
    "--experiment_name exp-1/char_r_ --model lstm --add_row_id"
    "--experiment_name exp-1/char_lr_ --model lstm --add_label_tokens"
    "--experiment_name exp-1/char_se_r_ --model lstm --add_start_end_tokens --add_row_id"
    "--experiment_name exp-1/char_se_lr_ --model lstm --add_start_end_tokens --add_label_tokens"
    "--experiment_name exp-1/char_r_lr_ --model lstm --add_row_id --add_label_tokens"
    "--experiment_name exp-1/char_se_r_lr_ --model lstm --add_start_end_tokens --add_row_id --add_label_tokens"
    
    # Word models - Transformer
    "--experiment_name exp-1/word_ --tokenization word"
    "--experiment_name exp-1/word_se_ --tokenization word --add_start_end_tokens"
    "--experiment_name exp-1/word_r_ --tokenization word --add_row_id"
    "--experiment_name exp-1/word_lr_ --tokenization word --add_label_tokens"
    "--experiment_name exp-1/word_se_r_ --tokenization word --add_start_end_tokens --add_row_id"
    "--experiment_name exp-1/word_se_lr_ --tokenization word --add_start_end_tokens --add_label_tokens"
    "--experiment_name exp-1/word_r_lr_ --tokenization word --add_row_id --add_label_tokens"
    "--experiment_name exp-1/word_se_r_lr_ --tokenization word --add_start_end_tokens --add_row_id --add_label_tokens"
    
    # Word models - LSTM
    "--experiment_name exp-1/word_ --tokenization word --model lstm"
    "--experiment_name exp-1/word_se_ --tokenization word --model lstm --add_start_end_tokens"
    "--experiment_name exp-1/word_r_ --tokenization word --model lstm --add_row_id"
    "--experiment_name exp-1/word_lr_ --tokenization word --model lstm --add_label_tokens"
    "--experiment_name exp-1/word_se_r_ --tokenization word --model lstm --add_start_end_tokens --add_row_id"
    "--experiment_name exp-1/word_se_lr_ --tokenization word --model lstm --add_start_end_tokens --add_label_tokens"
    "--experiment_name exp-1/word_r_lr_ --tokenization word --model lstm --add_row_id --add_label_tokens"
    "--experiment_name exp-1/word_se_r_lr_ --tokenization word --model lstm --add_start_end_tokens --add_row_id --add_label_tokens"
)

# Get the current configuration
current_config=${configs[$SLURM_ARRAY_TASK_ID]}

echo "Running experiment: $current_config"

# Run the experiment for the specified number of iterations
for i in $(seq 1 $ITERATIONS); do
    echo "Running iteration $i"
    python src/train.py $current_config
done