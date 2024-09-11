import pandas as pd
import argparse
import os
import json

from utils import setup_logger

def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Extract data from an experiment directory.")
    parser.add_argument("--experiment_name", type=str, default="exp-1", help="Experiment name")
    parser.add_argument("--output_file", type=str, default="results.csv", help="Output file")
    return parser.parse_args(args)

def check_folder(folder_path):
    if not os.path.exists(folder_path):
        log.error(f"Folder {folder_path} does not exist.")
        return False
    if not os.path.isdir(folder_path):
        log.error(f"Folder {folder_path} is not a directory.")
        return False
    if not os.path.exists(os.path.join(folder_path, "results.json")):
        log.error(f"File results.json does not exist in folder {folder_path}.")
        return False
    if not os.path.exists(os.path.join(folder_path, "args.json")):
        log.error(f"File args.json does not exist in folder {folder_path}.")
        return False
    return True

def extract_data(data_path):
    folder_paths = [
        os.path.join(data_path, folder) 
        for folder in os.listdir(data_path) 
        if check_folder(os.path.join(data_path, folder))
    ]
    data = []
    for folder_path in folder_paths:
        with open(os.path.join(folder_path, "results.json"), "r") as f:
            results = json.load(f)
        with open(os.path.join(folder_path, "args.json"), "r") as f:
            experiment_args = json.load(f)
        data.append({
            **experiment_args,
            **results,
            "folder_path": folder_path,
        })
    
    df = pd.DataFrame(data)
    output_file = os.path.join(data_path, args.output_file)
    df.to_csv(output_file, index=False)

if __name__ == "__main__":
    args = parse_args()
    data_path = os.path.join('results', args.experiment_name)
    log = setup_logger(os.path.join(data_path, "extract.log"), __name__)
    extract_data(data_path)