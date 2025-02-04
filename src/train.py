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

from datapipeline import create_sequences_token, load_data, create_sequences, prepare_datasets, load_table_lock_data
from model import build_lstm_model, build_transformer_model
from utils import setup_logger, is_table_locks
from evaluate import evaluate_predictions, print_examples, evaluate_naive_baseline, detokenization

def parse_args(args=None):
    if isinstance(args, dict): # For debugging
        # Return namespace from dict
        return argparse.Namespace(**args)

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

    parser.add_argument("--shuffle", action="store_true", default=False, help="Shuffle entire dataset. Not recommended since we want to preserve sequence order.")
    parser.add_argument("--add_start_end_tokens", action="store_true", default=False, help="Add start and end tokens")
    parser.add_argument("--add_row_id", action="store_true", default=False, help="Add row id")
    parser.add_argument("--add_label_tokens", action="store_true", default=False, help="Add label tokens")
    parser.add_argument("--disable_train_shuffle", action="store_true", default=False, help="Disable shuffling training data")
    parser.add_argument("--early_stopping", action="store_true", default=False, help="Use early stopping. Not recommended for comparision since different runs will stop at different epochs.")
    parser.add_argument("--remove_system_tables", action="store_true", default=False, help="Remove system tables from the dataset")
    parser.add_argument("--token_length_seq", action="store_true", default=False, help="Use token length in order to create sequences")
    parser.add_argument("--lstm_pe", action="store_true", default=False, help="Use position embedding in LSTM model")
    parser.add_argument("--naive_baseline", action="store_true", default=False, help="Use naive baseline")
    parser.add_argument("--disable_cache", action="store_true", default=False, help="Disable caching")

    parser.add_argument("--args_file", type=str, default=None, help="Load args from a json file. This will override all other args.")

    parsed_args = parser.parse_args(args)

    if parsed_args.args_file:
        with open(parsed_args.args_file, "r") as f:
            args_dict = json.load(f)
        parsed_args = argparse.Namespace(**args_dict)

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
        )
    else:
        data = load_data(
            data=df.copy(),
            char_based=char_based,
            add_row_id=args.add_row_id,
            add_start_end_tokens=args.add_start_end_tokens,
            add_label_tokens=args.add_label_tokens,
            remove_system_tables=args.remove_system_tables,
        )

    log.info("Computing vocab and output size...")

    num_unqiue_table_names = len(data["TABNAME"].unique())
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
    vocab_size += 1 # Add 1 to the vocab size to account for the temporary <OOV> token
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
        args_dict = dict(args.__dict__)
        log.warning("model_weights and args_file are removed from the args hash.")
        args_dict.pop("model_weights", None)
        args_dict.pop("args_file", None)
        args_hash = hashlib.sha256(json.dumps(args_dict).encode('utf-8')).hexdigest()
        # create the dir if it doesn't exist
        os.makedirs("data/.cache", exist_ok=True)
        # check if cache already exists
        if os.path.exists(f"data/.cache/cached_sequences_{args_hash}.pkl") and not args.disable_cache:
            log.info(f"Loading cached sequences for args: {args_hash}")
            with open(f"data/.cache/cached_sequences_{args_hash}.pkl", "rb") as f:
                source_texts, target_texts = pickle.load(f)
        else:
            log.info(f"Creating sequences for args: {args_hash}")
            source_texts, target_texts = create_sequences_token(data, args.seq_length, args.horizon) 
            # cache the sequences using the hash as the file name
            with open(f"data/.cache/cached_sequences_{args_hash}.pkl", "wb") as f:
                pickle.dump((source_texts, target_texts), f)
    else:
        source_texts, target_texts = create_sequences(data, args.seq_length) 
    x_train, x_test, y_train, y_test, source_tokenizer, target_tokenizer = prepare_datasets(
        source_texts,
        target_texts,
        vocab_size,
        args.seq_length,
        out_seq_length,
        args.test_split,
        args.shuffle,
    )

    if args.train_data_percent_used < 1.0:
        x_train = x_train[:int(len(x_train) * args.train_data_percent_used)]
        y_train = y_train[:int(len(y_train) * args.train_data_percent_used)]

    log.info(f"x_train shape: {x_train.shape}")
    log.info(f"y_train shape: {y_train.shape}")
    log.info(f"x_test shape: {x_test.shape}")
    log.info(f"y_test shape: {y_test.shape}")

    if args.naive_baseline:
        log.info("Evaluating naive baseline...")
        y_test_argmax = np.argmax(y_test, axis=-1)

        if len(y_test_argmax.shape) == 1:
            y_test_argmax = np.expand_dims(y_test_argmax, axis=-1)

        results, predictions, actual_values = evaluate_naive_baseline(y_test_argmax)
        log.info(f"Naive Baseline Results:\n{json.dumps(results, indent=4)}")
        # Save results to a file
        with open(os.path.join(results_folder_path, "results.json"), "w") as f:
            json.dump(results, f, indent=4)
        # Save args to a file
        with open(os.path.join(results_folder_path, "args.json"), "w") as f:
            json.dump(args.__dict__, f, indent=4)

        out_lock_preds, in_lock_sequences, gt_lock = detokenization(predictions, x_test, actual_values, target_tokenizer, source_tokenizer)

        df_out = pd.DataFrame({
            "in_lock_sequences": in_lock_sequences[:-1], # remove the last since we don't have a prediction for it
            "out_lock_preds": out_lock_preds,
            "gt_lock": gt_lock,
        })

        # Save lock sequences and predictions
        with open(os.path.join(results_folder_path, "predictions.csv"), "w") as f:
            df_out.to_csv(f, index=False)

        exit()

    log.info("Building model...")
    if args.model == "transformer":
        model = build_transformer_model(vocab_size, args.seq_length, out_seq_length)
    elif args.model == "lstm":
        model = build_lstm_model(vocab_size, args.seq_length, out_seq_length, position_embedding=args.lstm_pe)
    else:
        raise ValueError(f"Model {args.model} not found")

    # Compile the model
    opt = keras.optimizers.AdamW(learning_rate=args.learning_rate)
    model.compile(optimizer=opt,
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    # Display model summary
    model.summary(print_fn=log.info)

    # Define the EarlyStopping callback
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=args.patience, restore_best_weights=True
    )

    callbacks = []
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
            epochs=args.epochs,
            batch_size=args.batch_size,
            validation_split=args.val_split,
            callbacks=callbacks,
            shuffle=(not args.disable_train_shuffle)
        )

    log.info("Evaluating model...")

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
    })

    results = evaluate_predictions(y_pred_argmax, x_test, y_test_argmax, args.tokenization, args.horizon)

    results["loss"] = float(loss)
    results["accuracy_per_output"] = accuracy

    log.info("Saving results...")
    # Save results to a file
    with open(os.path.join(results_folder_path, "results.json"), "w") as f:
        json.dump(results, f, indent=4)

    # Save args to a file
    with open(os.path.join(results_folder_path, "args.json"), "w") as f:
        json.dump(args.__dict__, f, indent=4)

    # Save model
    model.save(os.path.join(results_folder_path, "model.keras"))

    if not args.model_weights:
        # Save history
        with open(os.path.join(results_folder_path, "history.json"), "w") as f:
            json.dump(history.history, f, indent=4)

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
