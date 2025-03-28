import numpy as np
import os
import logging

log = logging.getLogger(__name__)

def detokenization(y_pred, x, y, target_tokenizer, source_tokenizer):
    # Detokenize the predicted, input, and expected output sequences
    y_pred_detokenized = target_tokenizer.sequences_to_texts(y_pred)
    x_detokenized = source_tokenizer.sequences_to_texts(x)
    y_detokenized = target_tokenizer.sequences_to_texts(y)
    return y_pred_detokenized, x_detokenized, y_detokenized

def print_examples(y_pred, x, y, target_tokenizer, source_tokenizer):
    # Predict on a few examples
    log.info("Predicting a few examples...")
    for i in range(15):
        log.info(f"Input text: {source_tokenizer.sequences_to_texts([x[i]])}")
        log.info(f"Expected output text: {target_tokenizer.sequences_to_texts([y[i]])}")
        log.info(f"Predicted output text: {target_tokenizer.sequences_to_texts([y_pred[i]])}")

def evaluate_predictions(y_pred, x, y, tokenization_type, horizon=1):  
    # Calculate the actual test accuracy
    actual_test_accuracy_value = np.sum(np.all(y == y_pred, axis=-1))/len(y)
    log.info(f"Actual Test Accuracy (n={len(x)}): {actual_test_accuracy_value * 100:.2f}%")

    if horizon > 1:
        log.warning("Horizon > 1 is not supported for task specific accuracy calculation.")
        return {
            "actual_test_accuracy": actual_test_accuracy_value,
            "table_name_test_accuracy": None,
            "pageid_test_accuracy": None,
            "padding_test_accuracy": None,
        }

    # Calculate the task specific accuracy for table name, pageid, and padding
    count_table_name = 0
    count_pageid = 0
    count_padding = 0
    for i in range(len(x)):
        if tokenization_type == "char":
            if y[i][0] == y_pred[i][0]: # the first output token is always the table name
                count_table_name += 1
            # Get the index of the first padding token from the true ylabel
            padding_index = np.argmax(y[i] == 0) # argmax returns the first occurence
            if padding_index > 0:
                # Use the padding index to slice the pageid and padding from the predicted output
                if np.all(y[i][1:padding_index] == y_pred[i][1:padding_index]):
                    count_pageid += 1
                if np.all(y[i][padding_index:] == y_pred[i][padding_index:]):
                    count_padding += 1
            # Else if there are no padding tokens in the true ylabel
            else:
                count_padding += 1 # Count the padding as correct if there is no padding
                if np.all(y[i][1:] == y_pred[i][1:]):
                    count_pageid += 1
        elif tokenization_type == "word":
            if y[i][0] == y_pred[i][0]:
                count_table_name += 1
            if y[i][1] == y_pred[i][1]: # the second output token is the pageid in word tokenization
                count_pageid += 1
            count_padding = None # Word tokenization does not have padding in the output
        else:
            raise ValueError(f"Invalid tokenization type: {tokenization_type}")
        
    table_name_test_accuracy = count_table_name/len(x)
    pageid_test_accuracy = count_pageid/len(x)
    if count_padding is not None:
        padding_test_accuracy = count_padding/len(x)
    else:
        padding_test_accuracy = None

    log.info(f"Table Name Test Accuracy: {table_name_test_accuracy * 100:.2f}%")
    log.info(f"Page ID Test Accuracy: {pageid_test_accuracy * 100:.2f}%")
    if count_padding is not None:
        log.info(f"Padding Test Accuracy: {padding_test_accuracy * 100:.2f}%")

    results = {
        "actual_test_accuracy": actual_test_accuracy_value,
        "table_name_test_accuracy": table_name_test_accuracy,
        "pageid_test_accuracy": pageid_test_accuracy,
        "padding_test_accuracy": padding_test_accuracy,
    }
    return results

def evaluate_naive_baseline(y_test):
    # Generate predictions by assuming the next value is the same as the current one
    # y_test should already be prepared with the correct horizon
    y_pred = y_test.copy()[:-1]
    y_test = y_test[1:]
    # Calculate the number of correct predictions
    correct_predictions = sum(np.all(y_test == y_pred, axis=-1))

    # Compute accuracy
    accuracy = correct_predictions / len(y_test)
    return {
            "actual_test_accuracy": accuracy,
            "table_name_test_accuracy": None,
            "pageid_test_accuracy": None,
            "padding_test_accuracy": None,
        }, y_pred, y_test

# This naive baseline implementation is robust to scenarios where the next
# lock isn't necessarily the previous ground truth lock.
# This can happen when during sequence creation, multiple locks fit within the
# Token length budget and thus are appended to the input sequence.
def evaluate_naive_baseline_skip(y_pred, y_test):
    # Calculate the number of correct predictions
    correct_predictions = sum(np.all(y_test == y_pred, axis=-1))

    # Compute accuracy
    accuracy = correct_predictions / len(y_test)
    return {
            "actual_test_accuracy": accuracy,
            "table_name_test_accuracy": None,
            "pageid_test_accuracy": None,
            "padding_test_accuracy": None,
        }, y_pred, y_test
