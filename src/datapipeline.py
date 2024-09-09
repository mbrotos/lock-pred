import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
import numpy as np

def load_data(file_path, char_based=True):
    data = pd.read_csv(file_path)

    # Strip spaces from column headers
    data.columns = data.columns.str.strip()

    # Create features
    data["page_table_combined"] = (
        (
            data["PAGEID"].astype(str).apply(lambda x: " ".join(x)) if char_based 
            else data["PAGEID"].astype(str)
        )
        + " "
        + data["TABNAME"].astype(str).apply(lambda x: x.replace("_", ""))
    )

    return data

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(
            data.iloc[i : i + seq_length][["page_table_combined"]]
            .apply(" ".join)
            .reset_index()
            .values[0][1]
        )
        y.append(
            data.iloc[i + seq_length]["page_table_combined"]
        )  # Predicting combined feature
    return X, y

def tokenize_data(text, vocab_size, max_length):
    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(text)
    source_sequences = tokenizer.texts_to_sequences(text)
    padded_source_sequences = pad_sequences(
        source_sequences, maxlen=max_length, padding="post", value=0.0
    )
    return padded_source_sequences, tokenizer

def split_data(input_data, output_data, test_size, shuffle=False):
    indices = np.arange(len(input_data))
    # Don't shuffle the data. Time leakage, see:
    # https://en.wikipedia.org/wiki/Leakage_(machine_learning)#:~:text=Non%2Di.i,origin%20cross%20validation
    if shuffle:
        np.random.shuffle(indices) 
    split_index = int(len(indices) * test_size)
    train_indices = indices[:split_index]
    test_indices = indices[split_index:]
    
    x_train, x_test = input_data[train_indices], input_data[test_indices]
    y_train, y_test = output_data[train_indices], output_data[test_indices]
    return x_train, x_test, y_train, y_test

def prepare_datasets(source_texts, target_texts, vocab_size, max_length, out_seq_length, test_size, shuffle=False):
    input_data, source_tokenizer = tokenize_data(source_texts, vocab_size, max_length)
    output_data, target_tokenizer = tokenize_data(target_texts, vocab_size, out_seq_length)
    output_data = to_categorical(output_data, num_classes=vocab_size)
    x_train, x_test, y_train, y_test = split_data(input_data, output_data, test_size, shuffle)
    return x_train, x_test, y_train, y_test, source_tokenizer, target_tokenizer
