# lock-pred

## Setup

Create a Python 3.11 virtual environment: See `0_prep.sh`. 


```4:6:0_prep.sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
*Note: For SLURM-based systems, see `requirements_slurm.txt`.*

## Project Structure

- `src/`: Contains the main source code
  - `train.py`: Main script for training models
  - `datapipeline.py`: Data loading and preprocessing
  - `model.py`: Model architectures (Transformer and LSTM)
  - `evaluate.py`: Evaluation metrics and result reporting
  - `utils.py`: Utility functions
  - `tests/`: Unit tests
- `experiments/`: Experiment scripts for different configurations
- `results/`: Directory to store experiment results

## Running Experiments

Several experiment scripts are provided in the `experiments/` directory. To run an experiment:

```
bash experiments/exp-2.sh
```

For SLURM-based systems, the same script can be run using the SLURM command:

```
sbatch experiments/exp-2.sh
```

*Note: Most experiment scripts are interopable with SLURM and normal bash execution.*

## Data Processing

After running experiments, use the extract and transform scripts to process the results. See the scripts for more details:

```
bash 2_extract.sh
bash 3_transform.sh
```

## Testing

Run unit tests using pytest command:

```
pytest
```