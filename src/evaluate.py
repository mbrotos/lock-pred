import numpy as np
import os
import logging

log = logging.getLogger(__name__)

def evaluate_model(y_pred, x_test, y_test, target_tokenizer, source_tokenizer, tokenization_type):
    # Predict on a few examples
    log.info("Predicting on a few examples")
    for i in range(15):
        # print("Input:", x_test[i])
        log.info(f"Input text: {source_tokenizer.sequences_to_texts([x_test[i]])}")
        # print("Expected output:", y_test[i])
        log.info(f"Expected output text: {target_tokenizer.sequences_to_texts([np.argmax(y_test[i], axis=-1)])}")
        output = y_pred[i]
        # print("Predicted output:", output)
        log.info(f"Predicted output text: {target_tokenizer.sequences_to_texts([np.argmax(output, axis=-1)])}")
    
    # calculate the accuracy using batching
    preds_all = np.argmax(y_pred, axis=-1)
    y_test_all = np.argmax(y_test, axis=-1)
    count = 0
    for i in range(len(x_test)):
        # TODO: Add heuristic to disregard padding tokens
        if np.all(y_test_all[i] == preds_all[i]):
            count += 1
    log.info(f"Actual Test Accuracy (n={len(x_test)}): {count/len(x_test) * 100:.2f}%")

    # Calculate the task specific accuracy for table name, pageid, and padding
    count_table_name = 0
    count_pageid = 0
    count_padding = 0
    for i in range(len(x_test)):
        if tokenization_type == "char":
            if y_test_all[i][0] == preds_all[i][0]: # the first output token is always the table name
                count_table_name += 1
            # Get the index of the first padding token from the true ylabel
            padding_index = np.argmax(y_test_all[i] == 0) # argmax returns the first occurence
            # Use the padding index to slice the pageid and padding from the predicted output
            if np.all(y_test_all[i][1:padding_index] == preds_all[i][1:padding_index]):
                count_pageid += 1
            if np.all(y_test_all[i][padding_index:] == preds_all[i][padding_index:]):
                count_padding += 1
        elif tokenization_type == "word":
            if y_test_all[i][0] == preds_all[i][0]:
                count_table_name += 1
            if y_test_all[i][1] == preds_all[i][1]: # the second output token is the pageid in word tokenization
                count_pageid += 1
            count_padding = None # Word tokenization does not have padding in the output
        else:
            raise ValueError(f"Invalid tokenization type: {tokenization_type}")

    log.info(f"Table Name Test Accuracy: {count_table_name/len(x_test) * 100:.2f}%")
    log.info(f"Page ID Test Accuracy: {count_pageid/len(x_test) * 100:.2f}%")
    if count_padding is not None:
        log.info(f"Padding Test Accuracy: {count_padding/len(x_test) * 100:.2f}%")
    results = {
        "actual_test_accuracy": count/len(x_test),
        "table_name_test_accuracy": count_table_name/len(x_test),
        "pageid_test_accuracy": count_pageid/len(x_test),
        "padding_test_accuracy": count_padding/len(x_test),
    }
    return results