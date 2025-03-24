#!/bin/bash

# NOTE: This experiment is used to test training a lstm model for each table separately.

# SLURM configurations (will be ignored if not running on SLURM)
#SBATCH --job-name=transform-exp-15
#SBATCH --account=def-miranska
#SBATCH --output=logs/transform-exp-15_%A_%a.out
#SBATCH --error=logs/transform-exp-15_%A_%a.err
#SBATCH --time=11:59:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=1
#SBATCH --mail-user=adam.sorrenti@torontomu.ca
#SBATCH --mail-type=ALL
#SBATCH --array=0-1
module load gcc arrow
# Activate virtual environment
source .venv/bin/activate

set -e  # Exit immediately if a command exits with a non-zero status
set -o pipefail  # Ensure pipeline errors are caught

# experiments=("results_old/fixed/exp-10" "results_old/fixed/exp-6-row-locks" "results_old/fixed/exp-6-table-locks" "results_old/fixed/row_sep/exp-11-row-locks" "results_old/fixed/row_sep/exp-11-row-locks-naive" "results_old/fixed/row_sep/exp-11-row-locks-row_id")
experiments=("exp-15-lstm-row-locks" "exp-15-lstm-table-locks")

experiment=${experiments[$SLURM_ARRAY_TASK_ID]}
echo "Extracting $experiment"

python src/extract.py \
    --experiment_name $experiment \
    --output_file "results.csv" # --skip_predictions

echo "Transforming $experiment"
python src/transform.py \
    --input_file "results/$experiment/results.csv" \
    --output_file "results_clean.csv"