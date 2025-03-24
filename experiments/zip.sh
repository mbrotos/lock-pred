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
#SBATCH --array=0-0
module load gcc arrow
# Activate virtual environment
source .venv/bin/activate

set -e  # Exit immediately if a command exits with a non-zero status
set -o pipefail  # Ensure pipeline errors are caught

#experiments=("exp-17-transformer-sorted-row-locks" "exp-17-transformer-sorted-table-locks" "exp-18-naive-sorted-row-locks" "exp-18-naive-sorted-table-locks" "exp-19-naive-sorted-row-locks" "exp-19-transformer-sorted-row-locks")
experiments=("exp-21-lstm-sorted-table-locks")
experiment=${experiments[$SLURM_ARRAY_TASK_ID]}
echo "Zipping $experiment"

zip -r "results/$experiment.zip" "results/$experiment"

