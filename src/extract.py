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
    parser.add_argument("--skip_predictions", action="store_true", default=False, help="Skip predictions extraction")
    parser.add_argument("--iterations", type=int, default=10, help="Maximum number of iterations to extract")
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

        if counter[args_key] > args.iterations:
            log.warning(f"More than {args.iterations} iterations for {args_key}, skipping...")
            continue

        
        # Save experiment summary info
        data.append({
            **experiment_args,
            **results,
            "folder_path": folder_path,
            "iteration": counter[args_key]
        })

        if args.skip_predictions:
            log.info(f"Skipping predictions extraction for {folder_path}")
            continue
        else:
            log.info(f"Extracting predictions for {folder_path}")

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
                # seperate predicitons into table and pageid by splitting the string by space, the first element is the table and the rest is the pageid
                pred_extracted = predictions['out_lock_preds'].str.split(' ', n=1, expand=True)
                if gt_extracted.isnull().any().any():
                    raise ValueError("Failed to extract lock from some gt_lock strings in horizon==1")
                predictions['gt_table'] = gt_extracted[0]
                predictions['gt_pageid'] = gt_extracted[1].str.replace(' ', '').astype(int)
                predictions['pred_table'] = pred_extracted[0]
                predictions['pred_pageid'] = pred_extracted[1].str.replace(' ', '')
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
                def get_match_lengths(match):
                    # match is a list of tuples, each tuple is a match for a lock
                    # E.g. [('orderline', '6 5 3 8 1 '), ('stock', '2 5 3 1 '), ('orderline', '6 5 3 8 1 '), ('stock', '7 7 3 7 7')]
                    # we should return a list of ints which is the length of each match
                    # were everything is split by space and counted
                    return [len(' '.join(m).split()) for m in match]
                
                def get_pred_matches(pred, lengths, pad_val='-1'):
                    # pred is a list of strings, each string is a prediction for a lock
                    # lengths is a list of ints, each int is the length of a match
                    # E.g. ['orderline', '6', '5', '3', '8', '1', 'stock', '2', '0', '2', '1', '1', 'orderline', '6', '5', '3', '8', '1', 'stock', '2', '2', '2', '1']
                    # E.g. [6, 6, 6, 5]
                    # we should return a list of tuples, each tuple is a prediction for a lock
                    # were the first element of the tuple is the table and the second is the pageid as a string with spaces between element
                    # E.g. [('orderline', '6 5 3 8 1 '), ('stock', '2 0 2 1 1 '), ('orderline', '6 5 3 8 1 '), ('stock', '2 2 2 1 ')]

                    total_required = sum(lengths)
                    # Pad the overall prediction list if it's too short
                    if len(pred) < total_required:
                        pred = pred + [pad_val] * (total_required - len(pred))
                    results = []
                    index = 0
                    for l in lengths:
                        # Extract the current match chunk
                        chunk = pred[index:index+l]
                        # The first element is the table name
                        table = chunk[0]
                        # The remaining elements form the pageid; join them with spaces and add a trailing space
                        pageid = ' '.join(chunk[1:]) + ' '
                        results.append((table, pageid))
                        index += l
                    return results


                predictions['gt_lock_matches'] = predictions['gt_lock'].str.findall(LOCK_REGEX)
                predictions['gt_lock_matches_len'] = predictions['gt_lock_matches'].apply(get_match_lengths)
                predictions['pred_lock_split'] = predictions['out_lock_preds'].str.split(' ')
                predictions['pred_lock_matches'] = predictions.apply(
                    lambda row: get_pred_matches(row['pred_lock_split'], row['gt_lock_matches_len']),
                    axis=1
                )
                if not (predictions['gt_lock_matches'].str.len() == horizon).all():
                    mismatch = predictions.index[predictions['gt_lock_matches'].str.len() != horizon].tolist()
                    raise ValueError(f"Mismatch: ground truth lock matches count != horizon for rows: {mismatch}")
                predictions['lock_pairs'] = predictions.apply(
                    lambda row: list(zip(row['gt_lock_matches'], row['pred_lock_matches'])),
                    axis=1
                )
                predictions.drop(columns=['gt_lock_matches', 'gt_lock_matches_len', 'pred_lock_split', 'pred_lock_matches'], inplace=True)

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
                    lambda pair: ''.join(pair[1][1].split()) if (pair[1] is not None and pair[1][1] is not None) else None
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
        log.info("Skipping predictions extraction, done!")
        return
    else:
        log.info("Saving predictions and in_lock_sequences map...")

    predictions_df.to_parquet(os.path.join(data_path, "predictions.parquet"))
    in_lock_sequences_df = pd.DataFrame(
        list(in_lock_sequences_map.items()),
        columns=['in_lock_sequences', 'in_lock_sequences_id']
    )
    in_lock_sequences_df.to_parquet(os.path.join(data_path, "in_lock_sequences.parquet"))
    log.info("Done!")

if __name__ == "__main__":
    args = parse_args()
    data_path = os.path.join('results', args.experiment_name)
    log = setup_logger(os.path.join(data_path, "extract.log"))
    extract_data(data_path)
