#!/bin/bash

# NOTE: This experiment is used to test training a lstm model for each table separately.

# SLURM configurations (will be ignored if not running on SLURM)
#SBATCH --job-name=zip
#SBATCH --account=def-miranska
#SBATCH --output=logs/zip_%A_%a.out
#SBATCH --error=logs/zip_%A_%a.err
#SBATCH --time=11:59:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --mail-user=adam.sorrenti@torontomu.ca
#SBATCH --mail-type=ALL
#SBATCH --array=0-1
module load gcc arrow
# Activate virtual environment
source .venv/bin/activate

set -e  # Exit immediately if a command exits with a non-zero status
set -o pipefail  # Ensure pipeline errors are caught

#experiments=("exp-17-transformer-sorted-row-locks" "exp-17-transformer-sorted-table-locks" "exp-18-naive-sorted-row-locks" "exp-18-naive-sorted-table-locks" "exp-19-naive-sorted-row-locks" "exp-19-transformer-sorted-row-locks")
#experiments=("exp-41-naive-rounded-cut-row-locks" "exp-41-naive-rounded-cut-table-locks" "exp-42-naive-local-rounded-cut-row-locks" "exp-47-naive-rounded-qcut-row-locks" "exp-47-naive-rounded-qcut-table-locks" "exp-48-naive-local-rounded-qcut-row-locks" "exp-39-tranformer-rounded-cut-row-locks" "exp-39-tranformer-rounded-cut-table-locks" "exp-40-lstm-rounded-cut-row-locks" "exp-40-lstm-rounded-cut-table-locks" "exp-43-lstm-local-rounded-cut" "exp-44-transformer-local-rounded-cut" "exp-45-tranformer-rounded-qcut-row-locks" "exp-45-tranformer-rounded-qcut-table-locks" "exp-46-lstm-rounded-qcut-row-locks" "exp-46-lstm-rounded-qcut-table-locks" "exp-49-lstm-local-rounded-qcut" "exp-50-transformer-local-rounded-qcut")
experiments=("exp-40-lstm-rounded-cut-row-locks_100" "exp-40-lstm-rounded-cut-table-locks_100")
experiment=${experiments[$SLURM_ARRAY_TASK_ID]}
echo "Zipping $experiment"

zip -r "results/$experiment.zip" "results/$experiment"

