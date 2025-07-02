# Lock Prediction for Zero-Downtime Database Encryption
[![arXiv](https://img.shields.io/badge/arXiv-2506.23985-b31b1b.svg)](https://arxiv.org/abs/2506.23985)

This repository contains supplementary material for capturing lock data from IBM Db2 and training models to predict a sequence of locks. For more information, refer to the [preprint](https://arxiv.org/abs/2506.23985).

## Setup

1. Create a Python 3.11 virtual environment: See `0_prep.sh`. 


    ```4:6:0_prep.sh
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```
    *Note: For the narval system, see `requirements_narval.txt`.*

2. Download and extract the data to the `data/` directory.
- The data archive can be downloaded [here](https://drive.google.com/file/d/1LOiRjv-yrqNgQryPc8XFVP8lRgmYeNyK/view?usp=sharing).

## Project Structure
- `.vscode/`: Holds debug configuration file specific to VS Code
- `.github/workflows`: Contains GitHub Actions workflows for continuous integration and automation
- `analysis/`: Directory for results analysis R scripts and notebooks
- `data/`: Stores all downloaded and extracted data for the project
- `experiments/`: Experiment runner scripts for different configurations (includes SLURM integration)
- `logs/`: Stores experiment log files
- `play_scripts/`: Contains utility data and modelling exploration notebooks
- `results/`: Directory to store experiment results
- `src/`: Contains the main source code
  - `train.py`: Main script for training models
  - `datapipeline.py`: Data loading and preprocessing
  - `model.py`: Model architectures (Transformer and LSTM)
  - `evaluate.py`: Evaluation metrics and result reporting
  - `utils.py`: Utility functions
  - `tests/`: Unit tests
- `workload-testing/` Directory includes the [HammerDB](https://www.hammerdb.com/) scripts to start TPC-C workloads

## TPC-C Workload Simulation
All file names referenced in this subsection are relative to the `workload-testing/` directory.
 - Enable the DB2 trace using the command in `traceCommands.txt`.
 - Start TPC-C [HammerDB](https://www.hammerdb.com/) workloads by using `hammerDBinit.bat`,  `hammerDBStep1.bat`, and `hammerDBStep2.bat`. 
 - Stop the DB2 trace using the command in `traceCommands.txt`.
 - Start trace preprocessing by running `db2trc flw -t`.
 - Extract the row and table locks using `lookup_locknames.py`. 

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


## Citation

If you use or study the code, please cite it as follows.

```bibtex
@article{rakha2025lockprediction,
  title={Lock Prediction for Zero-Downtime Database Encryption},
  author={Mohamed Sami Rakha and Adam Sorrenti and Greg Stager and Walid Rjaibi and Andriy Miranskyy},
  journal={arXiv preprint arXiv:2506.23985},
  year={2025},
  doi={10.48550/arXiv.2506.23985}
}
```
