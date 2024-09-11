import pandas as pd
import argparse
import os

from utils import setup_logger

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_filename", type=str, default='results-clean.csv')
    return parser.parse_args(args)

def clean_data(df):
    df = df.drop(columns=["folder_path"])
    df = (
        df.groupby(
            df.columns.drop(["actual_test_accuracy", "loss", "accuracy_per_output"]).tolist()
        )
        .agg(
            actual_test_accuracy_mean=("actual_test_accuracy", "mean"),
            actual_test_accuracy_std=("actual_test_accuracy", "std"),
            loss_mean=("loss", "mean"),
            loss_std=("loss", "std"),
            accuracy_per_output_mean=("accuracy_per_output", "mean"),
            accuracy_per_output_std=("accuracy_per_output", "std"),
        )
        .reset_index()
    )
    return df
    
if __name__ == "__main__":
    args = parse_args()
    log_file = os.path.join(os.path.dirname(args.input_file), 'transform.log')
    log = setup_logger(log_file, __name__)
    df = pd.read_csv(args.input_file)
    df_clean = clean_data(df.copy())
    output_file = os.path.join(os.path.dirname(args.input_file), args.output_filename)
    df_clean.to_csv(output_file, index=False)
    log.info(f"Cleaned data saved to {output_file}")
