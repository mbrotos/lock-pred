from keras.src.callbacks import EarlyStopping
import numpy as np
import argparse
import json
import keras

from datapipeline import load_data, create_sequences, prepare_datasets
from model import build_lstm_model, build_transformer_model

def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument("--model", type=str, default="transformer", help="Model to use")
    parser.add_argument("--data", type=str, default="data/row_locks.csv", help="Data to use")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--seq_length", type=int, default=50, help="Sequence length")
    # NOTE: This out_seq_length was chosen to accomodate 6 significant digits for char-based tokenization
    parser.add_argument("--out_seq_length", type=int, default=6, help="Output sequence length")
    parser.add_argument("--test_split", type=float, default=0.3, help="Test dataset proportion")
    parser.add_argument("--val_split", type=float, default=0.3, help="Validation dataset proportion")
    # NOTE: Vocab size is the number of unique table in the dataset + 10 for digits + 1 for padding    
    parser.add_argument("--vocab_size", type=int, default=5+10+1, help="Vocabulary size")
    parser.add_argument("--tokenization", type=str, default="char", help="Tokenization")
    parser.add_argument("--patience", type=int, default=5, help="Patience for early stopping")
    parser.add_argument("--shuffle", type=bool, default=False, help="Shuffle data. Not recommended since we want to preserve sequence order.")
    return parser.parse_args(args)

def evaluate_model(model, x_test, y_test, target_tokenizer):
    # Evaluate the model on the test dataset
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f"Per-output Test Accuracy: {accuracy * 100:.2f}%")
    
    # Predict on a five example
    print("Predicting on a five example")
    for i in range(5):
        print("Input:", x_test[i])
        print("Expected output:", y_test[i])
        print("Expected output text:", target_tokenizer.sequences_to_texts([np.argmax(y_test[i], axis=-1)]))
        output = model.predict(x_test[np.newaxis, i])
        print("Predicted output:", output)
        print("Predicted output text:", target_tokenizer.sequences_to_texts([np.argmax(output, axis=-1)[0]]))
    
    # calculate the accuracy using batching
    preds_all = model.predict(x_test)
    preds_all = np.argmax(preds_all, axis=-1)
    y_test_all = np.argmax(y_test, axis=-1)
    count = 0
    for i in range(len(x_test)):
        if np.all(y_test_all[i] == preds_all[i]):
            count += 1
    print(f"Actual Test Accuracy (n={len(x_test)}): {count/len(x_test) * 100:.2f}%")

def main():
    args = parse_args()
    # Print args dict with indent
    print(json.dumps(args.__dict__, indent=4))

    # TODO: Add checks for args given buisness logic

    # Load data
    char_based = args.tokenization == "char"
    data = load_data(args.data, char_based)
    source_texts, target_texts = create_sequences(data, args.seq_length)
    x_train, x_test, y_train, y_test, source_tokenizer, target_tokenizer = prepare_datasets(
        source_texts,
        target_texts,
        args.vocab_size,
        args.seq_length,
        args.out_seq_length,
        args.test_split,
        args.shuffle,
    )

    if args.model == "transformer":
        model = build_transformer_model(args.vocab_size, args.seq_length, args.out_seq_length)
    elif args.model == "lstm":
        model = build_lstm_model(args.vocab_size, args.seq_length, args.out_seq_length)
    else:
        raise ValueError(f"Model {args.model} not found")
    
    # Compile the model
    opt = keras.optimizers.AdamW(learning_rate=args.learning_rate)
    model.compile(optimizer=opt,
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    # Display model summary
    model.summary()

    # Define the EarlyStopping callback
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=args.patience, restore_best_weights=True
    )
    
    callbacks = [early_stopping]
    # Train the model
    history = model.fit(
        x_train,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=args.val_split,
        callbacks=callbacks,
    )
    # TODO: Results should be saved to a file

    # TODO: Spin off eval to a separate script
    evaluate_model(model, x_test, y_test, target_tokenizer)


if __name__ == "__main__":
    main()
