from keras.src.callbacks import EarlyStopping
import numpy as np
import argparse
import json
import keras
import pandas as pd
import os
import datetime
import uuid
import pickle
import hashlib
import tensorflow as tf
import keras_nlp

from datapipeline import create_sequences_token, load_data, create_sequences, prepare_datasets, load_table_lock_data
from model import build_lstm_model, build_transformer_model_classifier, build_transformer_model_casual
from utils import setup_logger, is_table_locks
from evaluate import evaluate_predictions, print_examples, evaluate_naive_baseline, detokenization, evaluate_naive_baseline_skip

def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument("--model", type=str, default="transformer", help="Model to use")
    parser.add_argument("--data", type=str, default="data/row_locks.csv", help="Data to use")
    parser.add_argument("--train_data_percent_used", type=float, default=1.0, 
                        help="Percentage of training data to use. 1.0 is the entire dataset. "
                             "The test dataset will not be changed. Data is first split into "
                             "train and test, then the oldest lock sequences are removed from "
                             "the end of the train dataset until the desired percentage is reached.")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--seq_length", type=int, default=50, help="Sequence length")
    parser.add_argument("--test_split", type=float, default=0.3, help="Test dataset proportion")
    parser.add_argument("--val_split", type=float, default=0.3, help="Validation dataset proportion")
    # NOTE: Vocab size is used for word tokenization when the vocab size is not predefined.
    parser.add_argument("--vocab_size", type=int, default=900, help="Vocabulary size")
    parser.add_argument("--tokenization", type=str, default="char", help="Tokenization")
    parser.add_argument("--patience", type=int, default=5, help="Patience for early stopping")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory to save results")
    parser.add_argument("--experiment_name", type=str, default="", help="Experiment name")
    parser.add_argument("--horizon", type=int, default=1, help="Horizon for forecasting. I.e. how many steps ahead to predict")
    parser.add_argument("--model_weights", type=str, default=None, help="Model weights to load")
    parser.add_argument("--sort_by", type=str, default=None, help="Defines how to sort the locks. If None, no sorting is done.")

    parser.add_argument("--shuffle", action="store_true", default=False, help="[DEPRECATED] Shuffle entire dataset. Not recommended since we want to preserve sequence order.")
    parser.add_argument("--add_start_end_tokens", action="store_true", default=False, help="Add start and end tokens")
    parser.add_argument("--add_row_id", action="store_true", default=False, help="Add row id")
    parser.add_argument("--add_label_tokens", action="store_true", default=False, help="Add label tokens")
    parser.add_argument("--disable_train_shuffle", action="store_true", default=False, help="[DEPRECATED] Disable shuffling training data")
    parser.add_argument("--early_stopping", action="store_true", default=False, help="Use early stopping. Not recommended for comparision since different runs will stop at different epochs.")
    parser.add_argument("--remove_system_tables", action="store_true", default=False, help="Remove system tables from the dataset")
    parser.add_argument("--token_length_seq", action="store_true", default=False, help="Use token length in order to create sequences")
    parser.add_argument("--lstm_pe", action="store_true", default=False, help="Use position embedding in LSTM model")
    parser.add_argument("--naive_baseline", action="store_true", default=False, help="Use naive baseline")
    parser.add_argument("--disable_cache", action="store_true", default=False, help="Disable caching")
    parser.add_argument("--shuffle_train", action="store_true", default=False, help="Shuffle the training data after sequences are created.")
    parser.add_argument("--checkpoint", action="store_true", default=False, help="Save model checkpoints")
    parser.add_argument("--save_times_exit", action="store_true", default=False, help="Save times and exit")

    parser.add_argument("--args_file", type=str, default=None, help="Load args from a json file. This will override all other args.")

    if args is None:
        parsed_args = parser.parse_args()
    elif isinstance(args, dict):  # For debugging
        # Merge provided args with default args
        default_args = vars(parser.parse_args([]))
        merged_args = {**default_args, **args}
        parsed_args = argparse.Namespace(**merged_args)
    else:
        parsed_args = parser.parse_args(args)

    if parsed_args.args_file:
        with open(parsed_args.args_file, "r") as f:
            args_dict = json.load(f)
        # Merge loaded args with default args
        default_args = vars(parser.parse_args([]))
        merged_args = {**default_args, **args_dict}
        parsed_args = argparse.Namespace(**merged_args)

    return parsed_args

def main(args=None):
    # Print args dict with indent
    log.info(f"Arguments:\n{json.dumps(args.__dict__, indent=4)}")

    if not args.disable_cache:
        log.warning("Caching is enabled.")

    # TODO: Add checks for args given buisness logic

    # Load data
    char_based = args.tokenization == "char"
    table_lock = is_table_locks(args.data)

    log.info(f"Loading data...")
    df = pd.read_csv(args.data)

    if table_lock:
        if args.add_row_id or args.add_label_tokens or args.add_start_end_tokens:
            raise ValueError("Row id, label tokens, and start end tokens are not supported for table lock data.")
        data = load_table_lock_data(
            data=df.copy(),
            remove_system_tables=args.remove_system_tables,
            sort_by=args.sort_by,
        )
    else:
        data = load_data(
            data=df.copy(),
            char_based=char_based,
            add_row_id=args.add_row_id,
            add_start_end_tokens=args.add_start_end_tokens,
            add_label_tokens=args.add_label_tokens,
            remove_system_tables=args.remove_system_tables,
            sort_by=args.sort_by,
        )

    num_unqiue_table_names = len(data["TABNAME"].unique())

    if args.model == "transformer_causal":
        locks = data["input"]
        lock_tokens = locks.str.cat(sep=" ").split(" ")
        lock_sequences = []
        for i in range(0, len(lock_tokens), args.seq_length):
            lock_sequences.append(" ".join(lock_tokens[i:i+args.seq_length]))
        lock_sequences = pd.Series(lock_sequences)
        lock_sequences = lock_sequences[lock_sequences.str.split(" ").apply(len) == args.seq_length]

        # append "<bos>" to the beginning of each sequence and remove the last token from each sequence
        lock_sequences_input = lock_sequences.apply(lambda x: "<bos> " + x).apply(lambda x: " ".join(x.split(" ")[:-1]))
        # to numpy array
        source_texts = lock_sequences_input.to_numpy()
        target_texts = lock_sequences.to_numpy()

        log.warning("Timing data is not yet supported for transformer_causal model.")
        source_times = []
        target_times = []

        vocab_size = (
            num_unqiue_table_names +
            10 + # 10 digits
            1 # <bos>
        )
        out_seq_length = args.seq_length
        # assert that the length of the input and output sequences are all equal to seq_length
        assert all([len(x.split(" ")) == args.seq_length for x in source_texts])
        assert all([len(x.split(" ")) == args.seq_length for x in target_texts])

        # 
    else:


        log.info("Computing vocab and output size...")

        ## Compute vocab size
        if table_lock:
            vocab_size = num_unqiue_table_names + 1 #padding
        elif char_based:
            vocab_size = sum([
                num_unqiue_table_names,
                10, # 10 digits
                1, # padding
            ])
        else:
            vocab_size = args.vocab_size # keep arg when a vocab size is not predefined
            if args.add_row_id:
                # Add 50 to the vocab size to account for the unique row ids
                # NOTE: This is a temporary fix to account for the row ids.
                # We need to find a more general solution to account for the row ids.
                # Since row ids in the word tokenization scheme are dataset dependent,
                # we cannot simply add a fixed number of tokens to the vocab size for
                # all datasets.
                vocab_size += 50
        if args.add_start_end_tokens:
            vocab_size += 2 # start and end tokens

        if args.add_label_tokens and not args.add_row_id:
            vocab_size += 1 # page id label token
        elif args.add_label_tokens and args.add_row_id:
            vocab_size += 2 # page id and row id label tokens

        ## Compute the output sequence length
        if table_lock:
            out_seq_length = args.horizon
        elif char_based:
            # Using data dataframe compute the number of significant digits in the page_id
            page_id_digits = len(data["PAGEID"].max())
            out_seq_length = sum([
                1, # the table name
                page_id_digits,
            ]) * args.horizon
        else:
            out_seq_length = 2
            if args.horizon > 1:
                raise NotImplementedError("Horizon > 1 is not supported for word tokenization yet.")

        log.info("Creating sequences...")
        if args.token_length_seq:
            # Hash the args
            # lets define args_dict as only those args that are used above to load the data and such
            # this way we can cache the sequences for different args that use the same data
            args_dict = {
                "data": args.data,
                "seq_length": args.seq_length,
                "horizon": args.horizon,
                "tokenization": args.tokenization,
                "remove_system_tables": args.remove_system_tables,
                "add_row_id": args.add_row_id,
                "add_start_end_tokens": args.add_start_end_tokens,
                "add_label_tokens": args.add_label_tokens,
                "disable_cache": args.disable_cache,
                "token_length_seq": args.token_length_seq,
                "vocab_size": args.vocab_size,
                "sort_by": args.sort_by,
                "naive_baseline": args.naive_baseline,
            } 

            args_hash = hashlib.sha256(json.dumps(args_dict).encode('utf-8')).hexdigest()
            # create the dir if it doesn't exist
            os.makedirs("data/.cache", exist_ok=True)
            # check if cache already exists
            if os.path.exists(f"data/.cache/cached_sequences_{args_hash}.pkl") and not args.disable_cache:
                log.info(f"Loading cached sequences for args: {args_hash}")
                with open(f"data/.cache/cached_sequences_{args_hash}.pkl", "rb") as f:
                    source_texts, target_texts, naive_target_texts = pickle.load(f)
                with open(f"data/.cache/cached_times_{args_hash}.pkl", "rb") as f:
                    source_times, target_times = pickle.load(f)
            else:
                log.info(f"Creating sequences for args: {args_hash}")
                source_texts, target_texts, naive_target_texts, source_times, target_times = create_sequences_token(data, args.seq_length, args.horizon, args.naive_baseline) 
                # cache the sequences using the hash as the file name
                with open(f"data/.cache/cached_sequences_{args_hash}.pkl", "wb") as f:
                    pickle.dump((source_texts, target_texts, naive_target_texts), f)
                with open(f"data/.cache/cached_times_{args_hash}.pkl", "wb") as f:
                    pickle.dump((source_times, target_times), f)
        else:
            source_texts, target_texts = create_sequences(data, args.seq_length) 

    # check vocab size by counting unique tokens of source_texts
    unique_tokens = len(set(" ".join(source_texts).split(" "))) +1 # add 1 for padding
    #assert unique_tokens == vocab_size, f"Unique tokens: {len(unique_tokens)}, Vocab size: {vocab_size}"

    x_train, x_test, y_train, _, y_test, y_test_naive, source_tokenizer, target_tokenizer = prepare_datasets(
        source_texts,
        target_texts,
        naive_target_texts,
        vocab_size,
        args.seq_length,
        out_seq_length,
        args.test_split,
        False, # we don't want to shuffle the test data
        is_casual=(args.model == "transformer_causal"),
    )

    num_test = len(x_test)
    x_test_time = source_times[-num_test:]
    y_test_time = target_times[-num_test:]

    if args.save_times_exit:
        predictions_path = os.path.join(args.save_times_exit, "predictions.csv")
        predictions_df = pd.read_csv(predictions_path)
        # if x_test_time is larger than predictions
        if len(predictions_df) < len(x_test_time):
            x_test_time = x_test_time[:-1]
            y_test_time = y_test_time[:-1]

        predictions_df['in_lock_start_time'] = [t[0] for t in x_test_time]
        predictions_df['in_lock_end_time'] = [t[1] for t in x_test_time]
        predictions_df['gt_lock_start_time'] = [t[0] for t in y_test_time]
        predictions_df['gt_lock_end_time'] = [t[1] for t in y_test_time]
        predictions_df.to_csv(predictions_path, index=False)
        log.info(f"Saved times to {predictions_path}")
        exit()

    if args.train_data_percent_used < 1.0:
        x_train = x_train[:int(len(x_train) * args.train_data_percent_used)]
        y_train = y_train[:int(len(y_train) * args.train_data_percent_used)]

    # lets split off the validation data from the training data
    if args.val_split > 0:
        split_idx = int(len(x_train) * (1 - args.val_split))
        x_train, x_val = x_train[:split_idx], x_train[split_idx:]
        y_train, y_val = y_train[:split_idx], y_train[split_idx:]

    # Lets shuffle the training data
    if args.shuffle_train:
        log.info("Shuffling training sequences...")
        np.random.seed(42)
        idx = np.random.permutation(len(x_train))
        x_train = x_train[idx]
        y_train = y_train[idx]

    log.info(f"x_train shape: {x_train.shape}")
    log.info(f"y_train shape: {y_train.shape}")
    log.info(f"x_test shape: {x_test.shape}")
    log.info(f"y_test shape: {y_test.shape}")

    if args.naive_baseline:
        log.info("Evaluating naive baseline...")
        y_test_argmax = np.argmax(y_test, axis=-1)
        y_test_naive = np.argmax(y_test_naive, axis=-1)

        if len(y_test_argmax.shape) == 1:
            y_test_argmax = np.expand_dims(y_test_argmax, axis=-1)

        results, predictions, actual_values = evaluate_naive_baseline_skip(y_test_naive, y_test_argmax)
        log.info(f"Naive Baseline Results:\n{json.dumps(results, indent=4)}")
        # Save results to a file
        with open(os.path.join(results_folder_path, "results.json"), "w") as f:
            json.dump(results, f, indent=4)
        # Save args to a file
        with open(os.path.join(results_folder_path, "args.json"), "w") as f:
            json.dump(args.__dict__, f, indent=4)

        out_lock_preds, in_lock_sequences, gt_lock = detokenization(predictions, x_test, actual_values, target_tokenizer, source_tokenizer)

        df_out = pd.DataFrame({
            "in_lock_sequences": in_lock_sequences[1:], # remove the first lock since it is not predicted
            "out_lock_preds": out_lock_preds,
            "gt_lock": gt_lock,
            "in_lock_start_time": [t[0] for t in x_test_time][1:],
            "in_lock_end_time": [t[1] for t in x_test_time][1:],
            "gt_lock_start_time": [t[0] for t in y_test_time][1:],
            "gt_lock_end_time": [t[1] for t in y_test_time][1:],
        })

        # Save lock sequences and predictions
        with open(os.path.join(results_folder_path, "predictions.csv"), "w") as f:
            df_out.to_csv(f, index=False)

        exit()

    log.info("Building model...")
    if args.model == "transformer_causal":
        model = build_transformer_model_casual(vocab_size, args.seq_length)
        loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        mask_token_id = source_tokenizer.word_index["<bos>"]
        perplexity = keras_nlp.metrics.Perplexity(from_logits=True, mask_token_id=mask_token_id)
        opt = keras.optimizers.AdamW(learning_rate=args.learning_rate)
        model.compile(optimizer=opt, loss=loss_fn, metrics=[perplexity])
    elif args.model == "transformer":
        model = build_transformer_model_classifier(vocab_size, args.seq_length, out_seq_length)
        opt = keras.optimizers.AdamW(learning_rate=args.learning_rate)
        model.compile(optimizer=opt,
                    loss="categorical_crossentropy",
                    metrics=["accuracy"])
    elif args.model == "lstm":
        model = build_lstm_model(vocab_size, args.seq_length, out_seq_length, position_embedding=args.lstm_pe)
        opt = keras.optimizers.AdamW(learning_rate=args.learning_rate)
        model.compile(optimizer=opt,
                    loss="categorical_crossentropy",
                    metrics=["accuracy"])
    else:
        raise ValueError(f"Model {args.model} not found")

    # Display model summary
    model.summary(print_fn=log.info)

    # Define the EarlyStopping callback
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=args.patience, restore_best_weights=True
    )
    checkpoint = keras.callbacks.ModelCheckpoint(
        os.path.join(results_folder_path, "model.keras"),
        monitor="val_loss" if args.val_split > 0 else "loss",
        save_best_only=True,
    )

    callbacks = []
    if args.checkpoint:
        callbacks.append(checkpoint)
    if args.early_stopping:
        callbacks.append(early_stopping)
    
    if args.model_weights:
        # Load the model weights
        log.info(f"Loading model weights from {args.model_weights}")
        model.load_weights(args.model_weights)
    else:
        # Train the model
        log.info("Training model...")
        history = model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val) if args.val_split > 0 else None,
            epochs=args.epochs,
            batch_size=args.batch_size,
            callbacks=callbacks,
            shuffle=False
        )
        if not args.checkpoint:
            model.save(os.path.join(results_folder_path, "model.keras"))

    # Save args to a file
    with open(os.path.join(results_folder_path, "args.json"), "w") as f:
        json.dump(args.__dict__, f, indent=4)

    if not args.model_weights:
        # Save history
        with open(os.path.join(results_folder_path, "history.json"), "w") as f:
            json.dump(history.history, f, indent=4)

    with open(os.path.join(results_folder_path, "source_tokenizer.json"), "w") as f:
        json.dump(source_tokenizer.to_json(), f, indent=4)

    # load the best model
    model.load_weights(os.path.join(results_folder_path, "model.keras"))

    if args.model == "transformer_causal":
        target_tokenizer = source_tokenizer

        log.info("Evaluating causal model...")
        table_name_ids = [source_tokenizer.word_index[name.lower()] for name in data["TABNAME"].unique()]
        #stop_ids = table_name_ids + [source_tokenizer.word_index["<bos>"]]
        def next(prompt, cache, index):
            logits = model(prompt)[:, index-1 , :]
            # Ignore hidden states for now; only needed for contrastive search.
            hidden_states = None
            return logits, hidden_states, cache
        
        x_test_masked = x_test.copy()
        y_test_lst = []
        for i in range(len(x_test_masked)):
            lock_count = 0
            y_cur_table = []
            y_cur_pageid = [[] for _ in range(args.horizon)]
            y_cur_idx = (None, None)
            for j in range(len(x_test_masked[i]) - 1, 0, -1):
                if x_test_masked[i][j] in table_name_ids:
                    lock_count += 1
                    if lock_count == args.horizon + 1: # Add one to ignore the last lock since it may be truncated
                        y_cur_idx = (j, y_cur_idx[1])
                        y_cur_table.append(x_test_masked[i][j])
                        x_test_masked[i][j] = 0
                        break
                    elif lock_count > 1: 
                        y_cur_table.append(x_test_masked[i][j])
                elif lock_count > 0:
                    y_cur_pageid[lock_count - 1].append(x_test_masked[i][j])
                if lock_count == 1 and len(y_cur_pageid[lock_count - 1])==1:
                    y_cur_idx = (None, j)
                x_test_masked[i][j] = 0
                
            # Reverse the lists since we are going backwards
            y_cur_table = y_cur_table[::-1]
            y_cur_pageid = [y[::-1] for y in y_cur_pageid][::-1]
            # Zip the table and pageid lists and append to y_test_lst
            y_test_lst.append(list(zip(y_cur_table, y_cur_pageid))+[y_cur_idx])

        sampler = keras_nlp.samplers.GreedySampler()
        y_pred_argmax = []
        y_test_argmax = []
        for i in range(len(y_test_lst)):
            prompt_tokens = x_test_masked[i].reshape(1, -1)
            start_index = y_test_lst[i][-1][0]
            end_index = y_test_lst[i][-1][1]
            output_tokens = sampler(
                next=next,
                stop_token_ids=None, # We will process the entire sequence after inference.
                prompt=prompt_tokens,
                index=start_index,  # Start sampling immediately after the [BOS] token.
            )[0]
            y_pred = output_tokens[start_index:end_index+1].numpy()
            y_test = x_test[i][start_index:end_index+1]
            y_pred_argmax.append(y_pred)
            y_test_argmax.append(y_test)

            if i < 15:  
                input_text = source_tokenizer.sequences_to_texts([x_test[i]])[0]
                pred_text = target_tokenizer.sequences_to_texts([y_pred])[0]
                gt_text = target_tokenizer.sequences_to_texts([y_test])[0]
                log.info(f"Input: {input_text}")
                log.info(f"Prediction: {pred_text}")
                log.info(f"Ground Truth: {gt_text}")
                log.info("")
            # else:
            #     x_test_masked = x_test_masked[:16]
            #     break
        
        # Padd y_pred_argmax and y_test_argmax to the same length using the max length
        max_len = max([len(y) for y in y_pred_argmax] + [len(y) for y in y_test_argmax])
        y_pred_argmax = [np.pad(y, (0, max_len - len(y))) for y in y_pred_argmax]
        y_test_argmax = [np.pad(y, (0, max_len - len(y))) for y in y_test_argmax]

        # make into numpy arrays
        y_pred_argmax = np.array(y_pred_argmax)
        y_test_argmax = np.array(y_test_argmax)

        # Calculate % correct
        loss = max(history.history["val_loss"])
        accuracy = np.mean(y_pred_argmax == y_test_argmax)
        log.info(f"Per-output Test Accuracy: {accuracy * 100:.2f}%")

        x_test = x_test_masked
                
    else:
        log.info("Evaluating classifier model...")

        y_pred = model.predict(x_test)
        y_pred_argmax = np.argmax(y_pred, axis=-1)
        y_test_argmax = np.argmax(y_test, axis=-1)

        loss = np.mean(keras.losses.categorical_crossentropy(y_test, y_pred).numpy())
        accuracy = np.mean(y_pred_argmax == y_test_argmax)
        log.info(f"Per-output Test Accuracy: {accuracy * 100:.2f}%")

        if len(y_test_argmax.shape) == 1:
            # This is required because we squeeze the output of the model to remove the singleton dimension
            # FIXME: A solution would be to change the datapipeline to add singleton dimensions
            y_test_argmax = np.expand_dims(y_test_argmax, axis=-1)
            y_pred_argmax = np.expand_dims(y_pred_argmax, axis=-1)

    print_examples(y_pred_argmax, x_test, y_test_argmax, target_tokenizer, source_tokenizer)

    out_lock_preds, in_lock_sequences, gt_lock = detokenization(y_pred_argmax, x_test, y_test_argmax, target_tokenizer, source_tokenizer)

    df_out = pd.DataFrame({
        "in_lock_sequences": in_lock_sequences,
        "out_lock_preds": out_lock_preds,
        "gt_lock": gt_lock,
        "in_lock_start_time": [t[0] for t in x_test_time],
        "in_lock_end_time": [t[1] for t in x_test_time],
        "gt_lock_start_time": [t[0] for t in y_test_time],
        "gt_lock_end_time": [t[1] for t in y_test_time],
    })

    results = evaluate_predictions(y_pred_argmax, x_test, y_test_argmax, args.tokenization, args.horizon)

    results["loss"] = float(loss)
    results["accuracy_per_output"] = accuracy

    log.info("Saving results...")
    # Save results to a file
    with open(os.path.join(results_folder_path, "results.json"), "w") as f:
        json.dump(results, f, indent=4)

    # Save lock sequences and predictions
    with open(os.path.join(results_folder_path, "predictions.csv"), "w") as f:
        df_out.to_csv(f, index=False)

if __name__ == "__main__":
    args = parse_args()
    results_folder_name = f"{args.model}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4()}"
    if args.experiment_name:
        results_folder_name = f"{args.experiment_name}_{results_folder_name}"
    results_folder_path = os.path.join(args.results_dir, results_folder_name)
    os.makedirs(results_folder_path, exist_ok=True)
    log = setup_logger(os.path.join(results_folder_path, "train.log"))
    main(args)
