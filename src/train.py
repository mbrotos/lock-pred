from keras.src.callbacks import EarlyStopping
import numpy as np
import argparse
import json
import keras
import pandas as pd
import os
import datetime
import uuid

from datapipeline import load_data, create_sequences, prepare_datasets
from model import build_lstm_model, build_transformer_model
from utils import setup_logger

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

    parser.add_argument("--shuffle", action="store_true", default=False, help="Shuffle entire dataset. Not recommended since we want to preserve sequence order.")
    parser.add_argument("--add_start_end_tokens", action="store_true", default=False, help="Add start and end tokens")
    parser.add_argument("--add_row_id", action="store_true", default=False, help="Add row id")
    parser.add_argument("--add_label_tokens", action="store_true", default=False, help="Add label tokens")
    parser.add_argument("--disable_train_shuffle", action="store_true", default=False, help="Disable shuffling training data")
    parser.add_argument("--early_stopping", action="store_true", default=False, help="Use early stopping. Not recommended for comparision since different runs will stop at different epochs.")
    parser.add_argument("--remove_system_tables", action="store_true", default=False, help="Remove system tables from the dataset")
    return parser.parse_args(args)

def evaluate_model(model, x_test, y_test, target_tokenizer, source_tokenizer):
    # Evaluate the model on the test dataset
    loss, accuracy = model.evaluate(x_test, y_test)
    log.info(f"Per-output Test Accuracy: {accuracy * 100:.2f}%")
    
    # Predict on a few examples
    log.info("Predicting on a few examples")
    for i in range(15):
        # print("Input:", x_test[i])
        log.info(f"Input text: {source_tokenizer.sequences_to_texts([x_test[i]])}")
        # print("Expected output:", y_test[i])
        log.info(f"Expected output text: {target_tokenizer.sequences_to_texts([np.argmax(y_test[i], axis=-1)])}")
        output = model.predict(x_test[np.newaxis, i])
        # print("Predicted output:", output)
        log.info(f"Predicted output text: {target_tokenizer.sequences_to_texts([np.argmax(output, axis=-1)[0]])}")
    
    # calculate the accuracy using batching
    preds_all = model.predict(x_test)
    preds_all = np.argmax(preds_all, axis=-1)
    y_test_all = np.argmax(y_test, axis=-1)
    count = 0
    for i in range(len(x_test)):
        # TODO: Add heuristic to disregard padding tokens
        if np.all(y_test_all[i] == preds_all[i]):
            count += 1
    log.info(f"Actual Test Accuracy (n={len(x_test)}): {count/len(x_test) * 100:.2f}%")

    results = {
        "loss": loss,
        "accuracy_per_output": accuracy,
        "actual_test_accuracy": count/len(x_test),
    }
    return results

def main(args):
    # Print args dict with indent
    log.info(f"Arguments:\n{json.dumps(args.__dict__, indent=4)}")
    
    # TODO: Add checks for args given buisness logic

    # Load data
    char_based = args.tokenization == "char"

    data = load_data(
        file_path=args.data,
        char_based=char_based,
        add_row_id=args.add_row_id,
        add_start_end_tokens=args.add_start_end_tokens,
        add_label_tokens=args.add_label_tokens,
        remove_system_tables=args.remove_system_tables,
    )

    ## Compute vocab size
    if char_based:
        num_unqiue_table_names = len(data["TABNAME"].unique())
        vocab_size = sum([
            num_unqiue_table_names, 
            10, # 10 digits
            1, # padding
        ])
    else:
        vocab_size = args.vocab_size # keep arg when a vocab size is not predefined

    if args.add_start_end_tokens:
        vocab_size += 2 # start and end tokens

    ## Compute the output sequence length
    # Using data dataframe compute the number of significant digits in the page_id
    page_id_digits = len(data["PAGEID"].max())
    if char_based:
        out_seq_length = sum([
            1, # the table name
            page_id_digits,
        ])
    else:
        out_seq_length = 2

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

    if args.model == "transformer":
        model = build_transformer_model(vocab_size, args.seq_length, out_seq_length)
    elif args.model == "lstm":
        model = build_lstm_model(vocab_size, args.seq_length, out_seq_length)
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
    # Train the model
    history = model.fit(
        x_train,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=args.val_split,
        callbacks=callbacks,
        shuffle=(not args.disable_train_shuffle)
    )

    results = evaluate_model(model, x_test, y_test, target_tokenizer, source_tokenizer)

    # Save results to a file
    with open(os.path.join(results_folder_path, "results.json"), "w") as f:
        json.dump(results, f, indent=4)

    # Save args to a file
    with open(os.path.join(results_folder_path, "args.json"), "w") as f:
        json.dump(args.__dict__, f, indent=4)

    # Save model
    model.save(os.path.join(results_folder_path, "model.keras"))

    # Save history
    with open(os.path.join(results_folder_path, "history.json"), "w") as f:
        json.dump(history.history, f, indent=4)

if __name__ == "__main__":
    args = parse_args()
    results_folder_name = f"{args.model}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4()}"
    if args.experiment_name:
        results_folder_name = f"{args.experiment_name}_{results_folder_name}"
    results_folder_path = os.path.join(args.results_dir, results_folder_name)
    os.makedirs(results_folder_path, exist_ok=True)
    log = setup_logger(results_folder_path, __name__)
    main(args)
