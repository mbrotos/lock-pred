import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
import numpy as np
import logging
import tqdm

log = logging.getLogger(__name__)

def convert_to_iso(timestamp):
    """Converts timestamp format to ISO 8601."""
    parts = timestamp.split("-")  # Split by '-'
    date_part = "-".join(parts[:3])  # Keep first three elements as YYYY-MM-DD
    time_part = parts[3].replace(".", ":", 2)  # Replace only first two dots in the time part
    return f"{date_part}T{time_part}" # Make sure this is UTC!

def prep_columns(data, remove_system_tables, sort_by=None, table_lock=False, rounding_bin_size=None):
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
        data = data.sort_values(by="Start Unix Timestamp", ascending=True)
    elif sort_by=='start_time-dedupe' or sort_by=='start_time_tabname-dedupe' or sort_by=='start_time_pageid-dedupe':
        # Sort by start time and remove duplicates with the same start time if they have:
        # - The same TABNAME for table locks
        # - The same TABNAME and PAGEID for row locks
        data = data.sort_values(by="Start Unix Timestamp", ascending=True)
        if table_lock:
            assert sort_by != 'start_time_pageid-dedupe', "sort_by cannot be 'start_time_pageid-dedupe' for table locks"
            if sort_by=='start_time_tabname-dedupe':
                data = data.sort_values(by=["Start Unix Timestamp", "TABNAME"], ascending=True)
            
            data = data.drop_duplicates(subset=["Start Unix Timestamp", "TABNAME"], keep="first")
        else:
            if sort_by=='start_time_pageid-dedupe':
                data = data.sort_values(by=["Start Unix Timestamp", "PAGEID", "TABNAME"], ascending=True)
            elif sort_by=='start_time_tabname-dedupe':
                data = data.sort_values(by=["Start Unix Timestamp", "TABNAME", "PAGEID"], ascending=True)
            data = data.drop_duplicates(subset=["Start Unix Timestamp", "TABNAME", "PAGEID"], keep="first")
    elif sort_by!=None:
        raise ValueError(f"Unknown sort_by value: {sort_by}")
    
    if rounding_bin_size is not None:
        data["PAGEID_unrounded"] = data["PAGEID"] # Store the original PAGEID for later use
        # floor the pagesize to the nearest bin size
        data["PAGEID"] = np.floor(data["PAGEID"].astype(float) / rounding_bin_size).astype(int).astype(str)
        assert data["PAGEID"].astype(int).max() <= 9, "PAGEID max is greater than 9. Check the range of PAGEID values, the max should be 90000."

    return data

def load_table_lock_data(
    data,
    remove_system_tables=False,
    sort_by=None
):
    data = prep_columns(data, remove_system_tables, sort_by, table_lock=True)
    
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
    sort_by=None,
    rounding_bin_size=None,
):
    # Strip spaces from column headers
    data = prep_columns(data, remove_system_tables, sort_by, table_lock=False, rounding_bin_size=rounding_bin_size)
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

def create_sequences_token(data, token_length, horizon=1, naive_baseline=False):
    X, y, y_naive = [], [], []
    X_time, y_time = [], []
    # Flag to stop the loop when we reach the end of the data
    # We need a flag because len(data) != number of tokens
    # We accumulate tokens until we reach the token length limit
    # or the end of the data.
    done = False 

    token_counts = data["input"].apply(lambda x: len(x.split())).values
    assert all(token_counts <= token_length), "A lock has more tokens than the allowable token length."
    
    n = len(data)

    for i in tqdm.tqdm(range(n), desc="Creating sequences", unit="lock"):
        # Our lookahead has reached the end of the data, thus we stop
        if done: 
            break

        # I dont expect i + horizon to be greater than n because we are looking ahead
        # If it is, i think we have a bug because if it wasnt caught in the
        # lookahead than it may mean we have a single lock greater than the token length.
        assert (i + 1) + horizon <= n, "We may have a lock that is greater than the token length."

        accumulated_tokens = 0

        j = i
        while j < n and accumulated_tokens < token_length and not done:
            if (j + 1) + horizon > n:
                # The lookahead has exceed the end of the data when we account for the horizon
                # We need to stop here or we will have an index out of bounds error.
                done = True 
                break

            cur_lock_token_count = token_counts[j]
            if accumulated_tokens + cur_lock_token_count > token_length:
                break
            else:
                if (j + 1) + horizon == n:
                    # The lookahead has added the last new lock we can add from the data
                    # The next iteration will not add any new locks
                    # Let's flag this so we can stop the loop after this iteration
                    done = True

                accumulated_tokens += cur_lock_token_count
                j += 1

        X.append(" ".join(data.iloc[i:j]["input"].values))
        y.append(" ".join(data.iloc[j:j+horizon]["output"].values))
        if naive_baseline:
            naive_prediction = data.iloc[j-1-(horizon-1):j]["output"].values
            assert len(naive_prediction) == horizon, "Naive prediction length does not match horizon."
            y_naive.append(" ".join(naive_prediction))

        x_start = data.iloc[i]["Start Unix Timestamp"]
        x_end = data.iloc[j-1]["End Unix Timestamp"]
        X_time.append((x_start, x_end))

        y_start = data.iloc[j]["Start Unix Timestamp"]
        y_end = data.iloc[j+horizon-1]["End Unix Timestamp"]
        y_time.append((y_start, y_end))

    return X, y, y_naive, X_time, y_time

def tokenize_data(text, vocab_size, max_length, special_tokens=[]):
    # NOTE: The <> symbols are not included in the filters so we don't split on them.
    tokenizer = Tokenizer(num_words=vocab_size, filters='!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n')
    tokenizer.fit_on_texts(np.concatenate([text, special_tokens]))
    source_sequences = tokenizer.texts_to_sequences(text)
    padded_source_sequences = pad_sequences(
        source_sequences, maxlen=max_length, padding="post", value=0.0
    )
    return padded_source_sequences, tokenizer

def split_data(input_data, output_data, output_data_naive, test_size, shuffle=False):
    indices = np.arange(len(input_data))
    # Don't shuffle the data. Time leakage, see:
    # https://en.wikipedia.org/wiki/Leakage_(machine_learning)#:~:text=Non%2Di.i,origin%20cross%20validation
    if shuffle:
        np.random.shuffle(indices) 
    split_index = int(len(indices) * (1-test_size))
    train_indices = indices[:split_index]
    test_indices = indices[split_index:]
    
    x_train, x_test = input_data[train_indices], input_data[test_indices]
    if len(output_data_naive) > 0:
        y_train, y_train_naive, y_test, y_test_naive = output_data[train_indices], output_data_naive[train_indices], output_data[test_indices], output_data_naive[test_indices]
    else:
        y_train, y_train_naive, y_test, y_test_naive = output_data[train_indices], [], output_data[test_indices], []
    return x_train, x_test, y_train, y_train_naive, y_test, y_test_naive

def prepare_datasets(source_texts, target_texts, naive_target_texts, vocab_size, max_length, out_seq_length, test_size, shuffle=False, is_casual=False):
    input_data, source_tokenizer = tokenize_data(source_texts, vocab_size, max_length)
    output_data_naive = []
    if is_casual:
        target_tokenizer = None
        output_data = source_tokenizer.texts_to_sequences(target_texts)
        output_data = pad_sequences(output_data, maxlen=out_seq_length, padding="post", value=0.0)
        if len(naive_target_texts) > 0:
            output_data_naive = source_tokenizer.texts_to_sequences(naive_target_texts)
            output_data_naive = pad_sequences(output_data_naive, maxlen=out_seq_length, padding="post", value=0.0)
    else:
        output_data, target_tokenizer = tokenize_data(target_texts, vocab_size, out_seq_length)
        output_data = to_categorical(output_data, num_classes=vocab_size)
        if len(naive_target_texts) > 0:
            output_data_naive = target_tokenizer.texts_to_sequences(naive_target_texts)
            output_data_naive = pad_sequences(output_data_naive, maxlen=out_seq_length, padding="post", value=0.0)
            output_data_naive = to_categorical(output_data_naive, num_classes=vocab_size)

    x_train, x_test, y_train, y_train_naive, y_test, y_test_naive = split_data(input_data, output_data, output_data_naive, test_size, shuffle)
    return x_train, x_test, y_train, y_train_naive, y_test, y_test_naive, source_tokenizer, target_tokenizer
