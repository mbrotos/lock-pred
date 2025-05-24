#!/bin/bash

# SLURM configurations (will be ignored if not running on SLURM)
#SBATCH --job-name=transform-exp-32-transformer-sorted-row-locks
#SBATCH --account=def-miranska
#SBATCH --output=logs/transform-exp-32-transformer-sorted-row-locks_%A_%a.out
#SBATCH --error=logs/transform-exp-32-transformer-sorted-row-locks_%A_%a.err
#SBATCH --time=11:59:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=1
#SBATCH --mail-user=adam.sorrenti@torontomu.ca
#SBATCH --mail-type=ALL
#SBATCH --array=0-0
module load gcc arrow
module load python/3.11
source ~/.venv/bin/activate

set -e  # Exit immediately if a command exits with a non-zero status
set -o pipefail  # Ensure pipeline errors are caught

experiments=("exp-32-transformer-sorted-row-locks")

experiment=${experiments[$SLURM_ARRAY_TASK_ID]}
echo "Extracting $experiment"

python src/extract.py \
    --experiment_name $experiment \
    --output_file "results.csv" --iterations 10 # --skip_predictions 

echo "Transforming $experiment"
python src/transform.py \
    --input_file "results/$experiment/results.csv" \
    --output_file "results_clean.csv" --iterations 10