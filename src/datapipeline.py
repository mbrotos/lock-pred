import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
import numpy as np
import logging

log = logging.getLogger(__name__)

def load_data(
    data,
    char_based=True,
    add_row_id=False,
    add_start_end_tokens=False,
    add_label_tokens=False,
    remove_system_tables=False,
):
    # Strip spaces from column headers
    data.columns = data.columns.str.strip()
    if char_based:
        data["PAGEID"] = data["PAGEID"].astype(str).apply(lambda x: " ".join(x))
        data["ROWID"] = data["ROWID"].astype(str).apply(lambda x: " ".join(x))
    else:
        data["PAGEID"] = data["PAGEID"].astype(str)
        data["ROWID"] = data["ROWID"].astype(str)
    
    data["TABNAME"] = data["TABNAME"].astype(str).apply(lambda x: x.replace("_", ""))

    if remove_system_tables:
        data = data[data["TABSCHEMA"] != "SYSIBM"]

    # Create features
    data["input"] = (
        ("<START> " if add_start_end_tokens else "") +
        (
            ("<ROWID> " if add_label_tokens and add_row_id else "") + 
            ((data["ROWID"] + " ") if add_row_id else "")
        ) +
        (
            ("<PAGEID> " if add_label_tokens else "") +
            (data["PAGEID"] + " ") # Always add the PAGEID
        ) +
        data["TABNAME"] + # Always add the TABNAME
        (" <END>" if add_start_end_tokens else "")
    )

    data["output"] = (
        data["PAGEID"] + " " + data["TABNAME"]
    )

    return data

def create_sequences(data, seq_length):
    log.warning("This function is deprecated. Use create_sequences_token() instead.")
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(
            data.iloc[i : i + seq_length][["input"]]
            .apply(" ".join)
            .reset_index()
            .values[0][1]
        )
        y.append(
            data.iloc[i + seq_length]["output"]
        )  # Predicting combined feature
    return X, y

def create_sequences_token(data, token_length, horizon=1):
    # Get the output sequence for the given index
    output = lambda x: " ".join(
        data.iloc[x:x+horizon]["output"].values
    )

    X, y = [], []
    # Flag to stop the loop when we reach the end of the data
    # We need a flag because len(data) != number of tokens
    # We accumulate tokens until we reach the token length limit
    # or the end of the data.
    done = False 

    for i in range(len(data)-horizon):
        if done: break

        cur_x_seq = data.iloc[i]["input"].split(" ")
        assert len(cur_x_seq) <= token_length, "Token length is too small"

        j_end = None
        for j in range(i+1, len(data)-horizon):
            next_x_seq = data.iloc[j]["input"].split(" ")
            if len(next_x_seq) + len(cur_x_seq) <= token_length:
                cur_x_seq.extend(next_x_seq)
            else:
                j_end = j
                break

        X.append(" ".join(cur_x_seq))
        if j_end is None: # We reached the end of the data
            y.append(output(len(data)-horizon))
            # Stop us from creating more sequences by just padding the last one
            done = True 
        else: # We reached the token length limit
            y.append(output(j_end))

    return X, y

def tokenize_data(text, vocab_size, max_length):
    # NOTE: The <> symbols are not included in the filters so we don't split on them.
    tokenizer = Tokenizer(num_words=vocab_size, filters='!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n', oov_token="<OOV>")
    tokenizer.fit_on_texts(text)
    source_sequences = tokenizer.texts_to_sequences(text)
    tokenizer.oov_token = None # Remove the <OOV> token so padding tokens are not confused for <OOV>
    # NOTE: Padding tends to truncate from the front since the sequences are created from
    # based on the number of lock observations, and not the number of tokens.

    # TODO: Fix this my creating sequences with the max number of observations
    # so padding doesn't truncate from the front. This way we do not lose information
    # when we truncate the sequences from the front. See create_sequences().
    padded_source_sequences = pad_sequences(
        source_sequences, maxlen=max_length, padding="post", value=0.0
    )
    # Make sure we don't have any <OOV> tokens
    assert np.all(padded_source_sequences != tokenizer.word_index["<OOV>"])
    return padded_source_sequences, tokenizer

def split_data(input_data, output_data, test_size, shuffle=False):
    indices = np.arange(len(input_data))
    # Don't shuffle the data. Time leakage, see:
    # https://en.wikipedia.org/wiki/Leakage_(machine_learning)#:~:text=Non%2Di.i,origin%20cross%20validation
    if shuffle:
        np.random.shuffle(indices) 
    split_index = int(len(indices) * (1-test_size))
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
