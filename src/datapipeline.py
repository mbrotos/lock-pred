import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
import numpy as np
import logging

log = logging.getLogger(__name__)

def convert_to_iso(timestamp):
    """Converts timestamp format to ISO 8601."""
    parts = timestamp.split("-")  # Split by '-'
    date_part = "-".join(parts[:3])  # Keep first three elements as YYYY-MM-DD
    time_part = parts[3].replace(".", ":", 2)  # Replace only first two dots in the time part
    return f"{date_part}T{time_part}" # Make sure this is UTC!

def prep_columns(data, remove_system_tables, sort_by=None):
    data.columns = data.columns.str.strip()
    # Remove all trailing whitespace from TABNAME and TABSCHEMA
    # NOTE: This seems to be a problem in the table lock data.
    data["TABNAME"] = data["TABNAME"].astype(str).apply(lambda x: x.rstrip())
    data["TABSCHEMA"] = data["TABSCHEMA"].astype(str).apply(lambda x: x.rstrip())
    
    # Remove all underscores from TABNAME
    data["TABNAME"] = data["TABNAME"].astype(str).apply(lambda x: x.replace("_", ""))

    data["Start Timestamp ISO8601"] = data["Start Timestamp"].apply(convert_to_iso)
    data["End Timestamp ISO8601"] = data["End Timestamp"].apply(convert_to_iso)
    data["Start Unix Timestamp"] = data["Start Timestamp ISO8601"].apply(
        lambda x: np.datetime64(x, 'ns').astype('int')
    )
    data["End Unix Timestamp"] = data["End Timestamp ISO8601"].apply(
        lambda x: np.datetime64(x, 'ns').astype('int')
    )
    # NOTE: To convert unix timestamp back to ISO timestamp, use:
    # np.datetime64('1970-01-01T00:00:00Z') + np.timedelta64(unix_ns, 'ns')
    
    if remove_system_tables:
        data = data[data["TABSCHEMA"] != "SYSIBM"]

    if sort_by=='start_time':
        data = data.sort_values(by="Start Unix Timestamp")

    return data

def load_table_lock_data(
    data,
    remove_system_tables=False,
    sort_by=None
):
    data = prep_columns(data, remove_system_tables, sort_by)
    
    data["input"] = data["TABNAME"]
    data["output"] = data["TABNAME"]
    return data

def load_data(
    data,
    char_based=True,
    add_row_id=False,
    add_start_end_tokens=False,
    add_label_tokens=False,
    remove_system_tables=False,
    sort_by=None
):
    # Strip spaces from column headers
    data = prep_columns(data, remove_system_tables, sort_by)
    if char_based:
        data["PAGEID"] = data["PAGEID"].astype(str).apply(lambda x: " ".join(x))
        data["ROWID"] = data["ROWID"].astype(str).apply(lambda x: " ".join(x))
    else:
        data["PAGEID"] = data["PAGEID"].astype(str)
        data["ROWID"] = data["ROWID"].astype(str)

    # Create features
    data["input"] = (
        ("<START> " if add_start_end_tokens else "") +
        data["TABNAME"] + " " + # Always add the TABNAME
        (
            ("<ROWID> " if add_label_tokens and add_row_id else "") + 
            ((data["ROWID"] + " ") if add_row_id else "")
        ) +
        (
            ("<PAGEID> " if add_label_tokens else "") +
            (data["PAGEID"]) # Always add the PAGEID
        ) +
        (" <END>" if add_start_end_tokens else "")
    )

    data["output"] = (
        data["TABNAME"] + " " + data["PAGEID"]
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
    output_time = lambda x: (
        data.iloc[x:x+horizon]["Start Unix Timestamp"].values[0], 
        data.iloc[x:x+horizon]["End Unix Timestamp"].values[-1]
    )

    X, y = [], []
    X_time, y_time = [], []
    # Flag to stop the loop when we reach the end of the data
    # We need a flag because len(data) != number of tokens
    # We accumulate tokens until we reach the token length limit
    # or the end of the data.
    done = False 

    for i in range(len(data)-horizon):
        if done: break

        cur_x_seq = data.iloc[i]["input"].split(" ")
        start_time = data.iloc[i]["Start Unix Timestamp"]
        end_time = None
        assert len(cur_x_seq) <= token_length, "Token length is too small"

        j_end = None
        for j in range(i+1, len(data)-horizon):
            next_x_seq = data.iloc[j]["input"].split(" ")
            if len(next_x_seq) + len(cur_x_seq) <= token_length:
                cur_x_seq.extend(next_x_seq)
                end_time = data.iloc[j]["End Unix Timestamp"]
            else:
                j_end = j
                break

        X.append(" ".join(cur_x_seq))
        if j_end is None: # We reached the end of the data
            y.append(output(len(data)-horizon))
            y_start, y_end = output_time(len(data)-horizon)
            X_time.append((start_time, end_time))
            y_time.append((y_start, y_end))
            # Stop us from creating more sequences by just padding the last one
            done = True 
        else: # We reached the token length limit
            y.append(output(j_end))
            y_start, y_end = output_time(j_end)
            X_time.append((start_time, end_time))
            y_time.append((y_start, y_end))

    return X, y, X_time, y_time

def tokenize_data(text, vocab_size, max_length, special_tokens=[]):
    # NOTE: The <> symbols are not included in the filters so we don't split on them.
    tokenizer = Tokenizer(num_words=vocab_size, filters='!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n')
    tokenizer.fit_on_texts(np.concatenate([text, special_tokens]))
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
    split_index = int(len(indices) * (1-test_size))
    train_indices = indices[:split_index]
    test_indices = indices[split_index:]
    
    x_train, x_test = input_data[train_indices], input_data[test_indices]
    y_train, y_test = output_data[train_indices], output_data[test_indices]
    return x_train, x_test, y_train, y_test

def prepare_datasets(source_texts, target_texts, vocab_size, max_length, out_seq_length, test_size, shuffle=False, is_casual=False):
    input_data, source_tokenizer = tokenize_data(source_texts, vocab_size, max_length)
    if is_casual:
        target_tokenizer = None
        output_data = source_tokenizer.texts_to_sequences(target_texts)
        output_data = pad_sequences(output_data, maxlen=out_seq_length, padding="post", value=0.0)
    else:
        output_data, target_tokenizer = tokenize_data(target_texts, vocab_size, out_seq_length)
        output_data = to_categorical(output_data, num_classes=vocab_size)
    x_train, x_test, y_train, y_test = split_data(input_data, output_data, test_size, shuffle)
    return x_train, x_test, y_train, y_test, source_tokenizer, target_tokenizer
