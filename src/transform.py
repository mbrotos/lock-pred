import pandas as pd
import argparse
import os

from utils import setup_logger


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_filename", type=str, default="results-clean.csv")
    parser.add_argument("--iterations", type=int, required=False, 
                        help="Minimum number of samples per group. Groups with fewer samples are dropped, and groups with more are truncated.")
    return parser.parse_args(args)


def filter_and_truncate_groups(df, group_cols, iterations, logger=None):
    """
    For each group defined by group_cols, enforce that it has exactly
    'iterations' rows. If a group has fewer than 'iterations', it's dropped;
    if more, it's truncated (randomly sampled or otherwise).
    """
    # Group the data
    grouped = df.groupby(group_cols, dropna=False)
    
    # Container for the filtered groups
    filtered_groups = []
    
    # Go through each group
    for group_key, group_df in grouped:
        if len(group_df) < iterations:
            # Log or print if desired
            if logger:
                logger.warning(f"Group {group_key} has only {len(group_df)} rows, "
                               f"which is less than required {iterations}. Dropping it.")
            # Skip this group
            continue
        elif len(group_df) > iterations:
            # Truncate to the size of 'iterations'
            group_df = group_df.head(iterations)
            if logger:
                logger.info(f"Group {group_key} has {len(group_df)} rows, truncating to {iterations}.")
        
        filtered_groups.append(group_df)
    
    # Concatenate all accepted/truncated groups
    if filtered_groups:
        return pd.concat(filtered_groups, ignore_index=True)
    else:
        return pd.DataFrame(columns=df.columns)


def clean_data(df, iterations=None, logger=None):
    # Identify the columns used to group and compute aggregates
    # so we know which columns must remain for grouping
    agg_columns_to_drop = [
        "actual_test_accuracy",
        "loss",
        "accuracy_per_output",
        "table_name_test_accuracy",
        "pageid_test_accuracy",
        "padding_test_accuracy",
        "folder_path",
    ]
    group_cols = df.columns.drop(agg_columns_to_drop, errors='ignore').tolist()

    # If iterations is provided, enforce group sizes before aggregating
    if iterations is not None:
        df = filter_and_truncate_groups(df, group_cols, iterations, logger)

    # Define the aggregation dictionary with the original column names
    possible_agg_columns = {
        "actual_test_accuracy": {
            "actual_test_accuracy_mean": "mean",
            "actual_test_accuracy_std": "std",
        },
        "loss": {
            "loss_mean": "mean",
            "loss_std": "std",
        },
        "accuracy_per_output": {
            "accuracy_per_output_mean": "mean",
            "accuracy_per_output_std": "std",
        },
        "table_name_test_accuracy": {
            "table_name_test_accuracy_mean": "mean",
            "table_name_test_accuracy_std": "std",
        },
        "pageid_test_accuracy": {
            "pageid_test_accuracy_mean": "mean",
            "pageid_test_accuracy_std": "std",
        },
        "padding_test_accuracy": {
            "padding_test_accuracy_mean": "mean",
            "padding_test_accuracy_std": "std",
        },
    }

    # Dynamically build the aggregation dictionary for available columns
    agg_dict = {
        new_col: (col, func)
        for col, metrics in possible_agg_columns.items()
        if col in df.columns
        for new_col, func in metrics.items()
    }

    # Perform aggregation using the dynamically built dictionary
    df_agg = (
        df.groupby(group_cols, dropna=False)
        .agg(**agg_dict)
        .reset_index()
    )

    return df_agg


if __name__ == "__main__":
    args = parse_args()
    log_file = os.path.join(os.path.dirname(args.input_file), "transform.log")
    log = setup_logger(log_file)
    
    df = pd.read_csv(args.input_file)
    
    df_clean = clean_data(df.copy(), iterations=args.iterations, logger=log)
    
    output_file = os.path.join(os.path.dirname(args.input_file), args.output_filename)
    df_clean.to_csv(output_file, index=False)
    
    log.info(f"Cleaned data saved to {output_file}")
