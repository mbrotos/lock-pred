import pandas as pd
import argparse
import os
import json
import re
import tqdm
import uuid
from collections import Counter

from utils import setup_logger, is_table_locks

# Pre-compile the regex for non-table locks
PATTERN = r'([A-Za-z]+)\s*((?:\d+\s*)+)'
LOCK_REGEX = re.compile(PATTERN)

def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Extract data from an experiment directory.")
    parser.add_argument("--experiment_name", type=str, default="exp-1", help="Experiment name")
    parser.add_argument("--output_file", type=str, default="results.csv", help="Output file")
    parser.add_argument("--skip_predictions", action="store_true", help="Skip predictions extraction")
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
    predictions_df = None
    in_lock_sequences_map = {}
    counter = Counter()

    for folder_path in tqdm.tqdm(folder_paths):
        print(folder_path)
        with open(os.path.join(folder_path, "results.json"), "r") as f:
            results = json.load(f)
        with open(os.path.join(folder_path, "args.json"), "r") as f:
            experiment_args = json.load(f)

        # Using the experiment_args as a key to count the number of iterations, but first remove 'model_weights' key
        args_copy = experiment_args.copy()
        args_copy.pop('model_weights', None)
        args_key = json.dumps(args_copy, sort_keys=True)
        counter[args_key] += 1

        
        # Save experiment summary info
        data.append({
            **experiment_args,
            **results,
            "folder_path": folder_path,
            "iteration": counter[args_key]
        })

        if args.skip_predictions:
            continue

        # Read predictions CSV for this folder
        predictions = pd.read_csv(os.path.join(folder_path, "predictions.csv"))

        # Map in_lock_sequences to unique ids
        unique_inputs = predictions['in_lock_sequences'].unique()
        for inp in unique_inputs:
            if inp not in in_lock_sequences_map:
                in_lock_sequences_map[inp] = len(in_lock_sequences_map) + 1
        predictions['in_lock_sequences'] = predictions['in_lock_sequences'].apply(lambda x: in_lock_sequences_map[x])
        predictions = predictions.rename(columns={"in_lock_sequences": "in_lock_sequences_id"})

        # Add experiment arguments to predictions (so each row carries experiment metadata)
        for key, value in experiment_args.items():
            predictions[key] = value

        table_locks = is_table_locks(experiment_args['data'])
        horizon = experiment_args['horizon']

        if horizon == 1:
            # --- Horizon == 1: one lock per row ---
            if table_locks:
                tokens_gt = predictions['gt_lock'].str.split()
                tokens_pred = predictions['out_lock_preds'].str.split()
                if not (tokens_gt.str.len() == 1).all():
                    raise ValueError("Some gt_lock strings have != 1 token for table locks in horizon==1")
                if not (tokens_pred.str.len() == 1).all():
                    raise ValueError("Some out_lock_preds strings have != 1 token for table locks in horizon==1")
                predictions['gt_table'] = tokens_gt.str[0]
                predictions['gt_pageid'] = None
                predictions['pred_table'] = tokens_pred.str[0]
                predictions['pred_pageid'] = None
            else:
                # For non-table locks, use vectorized regex extraction.
                gt_extracted = predictions['gt_lock'].str.extract(PATTERN, expand=True)
                pred_extracted = predictions['out_lock_preds'].str.extract(PATTERN, expand=True)
                if gt_extracted.isnull().any().any():
                    raise ValueError("Failed to extract lock from some gt_lock strings in horizon==1")
                if pred_extracted.isnull().any().any():
                    raise ValueError("Failed to extract lock from some out_lock_preds strings in horizon==1")
                predictions['gt_table'] = gt_extracted[0]
                predictions['gt_pageid'] = gt_extracted[1].str.replace(' ', '').astype(int)
                predictions['pred_table'] = pred_extracted[0]
                predictions['pred_pageid'] = pred_extracted[1].str.replace(' ', '').astype(int)
            predictions['horizon_position'] = 1
            predictions['unique_id'] = [str(uuid.uuid4()) for _ in range(len(predictions))]

        else:
            # --- Horizon > 1: multiple locks per row, need to explode into multiple rows ---
            predictions['orig_idx'] = predictions.index
            predictions['unique_id'] = [str(uuid.uuid4()) for _ in range(len(predictions))]

            if table_locks:
                predictions['gt_tokens'] = predictions['gt_lock'].str.split()
                predictions['pred_tokens'] = predictions['out_lock_preds'].str.split()
                if not (predictions['gt_tokens'].str.len() == horizon).all():
                    mismatch = predictions.index[predictions['gt_tokens'].str.len() != horizon].tolist()
                    raise ValueError(f"Mismatch: ground truth tokens count != horizon for rows: {mismatch}")
                predictions['pred_tokens'] = predictions['pred_tokens'].apply(
                    lambda tokens: tokens + [None]*(horizon - len(tokens)) if len(tokens) < horizon else tokens[:horizon]
                )
                predictions['lock_pairs'] = predictions.apply(
                    lambda row: list(zip(row['gt_tokens'], row['pred_tokens'])),
                    axis=1
                )
            else:
                predictions['gt_lock_matches'] = predictions['gt_lock'].str.findall(LOCK_REGEX)
                predictions['pred_lock_matches'] = predictions['out_lock_preds'].str.findall(LOCK_REGEX)
                if not (predictions['gt_lock_matches'].str.len() == horizon).all():
                    mismatch = predictions.index[predictions['gt_lock_matches'].str.len() != horizon].tolist()
                    raise ValueError(f"Mismatch: ground truth lock matches count != horizon for rows: {mismatch}")
                predictions['pred_lock_matches'] = predictions['pred_lock_matches'].apply(
                    lambda matches: matches + [(None, None)]*(horizon - len(matches))
                    if len(matches) < horizon else matches[:horizon]
                )
                predictions['lock_pairs'] = predictions.apply(
                    lambda row: list(zip(row['gt_lock_matches'], row['pred_lock_matches'])),
                    axis=1
                )

            predictions = predictions.explode('lock_pairs').reset_index(drop=True)
            predictions['horizon_position'] = predictions.groupby('orig_idx').cumcount() + 1

            if table_locks:
                predictions['gt_table'] = predictions['lock_pairs'].apply(lambda pair: pair[0])
                predictions['gt_pageid'] = None
                predictions['pred_table'] = predictions['lock_pairs'].apply(lambda pair: pair[1])
                predictions['pred_pageid'] = None
            else:
                predictions['gt_table'] = predictions['lock_pairs'].apply(
                    lambda pair: pair[0][0] if (pair[0] is not None and pair[0][0] is not None) else None
                )
                predictions['gt_pageid'] = predictions['lock_pairs'].apply(
                    lambda pair: int(''.join(pair[0][1].split())) if (pair[0] is not None and pair[0][1] is not None) else None
                )
                predictions['pred_table'] = predictions['lock_pairs'].apply(
                    lambda pair: pair[1][0] if (pair[1] is not None and pair[1][0] is not None) else None
                )
                predictions['pred_pageid'] = predictions['lock_pairs'].apply(
                    lambda pair: int(''.join(pair[1][1].split())) if (pair[1] is not None and pair[1][1] is not None) else None
                )

            predictions = predictions.drop(
                columns=['gt_tokens', 'pred_tokens', 'gt_lock_matches', 'pred_lock_matches', 'lock_pairs', 'orig_idx'],
                errors='ignore'
            )

        predictions = predictions.drop(columns=['gt_lock', 'out_lock_preds'])

        # Add iteration number to predictions
        predictions['iteration'] = counter[args_key]

        if predictions_df is None:
            predictions_df = predictions
        else:
            predictions_df = pd.concat([predictions_df, predictions], ignore_index=True)

    df = pd.DataFrame(data)
    output_file = os.path.join(data_path, args.output_file)
    df.to_csv(output_file, index=False)

    if args.skip_predictions:
        return

    predictions_df.to_parquet(os.path.join(data_path, "predictions.parquet"))
    in_lock_sequences_df = pd.DataFrame(
        list(in_lock_sequences_map.items()),
        columns=['in_lock_sequences', 'in_lock_sequences_id']
    )
    in_lock_sequences_df.to_parquet(os.path.join(data_path, "in_lock_sequences.parquet"))

if __name__ == "__main__":
    args = parse_args()
    data_path = os.path.join('results', args.experiment_name)
    log = setup_logger(os.path.join(data_path, "extract.log"))
    extract_data(data_path)
