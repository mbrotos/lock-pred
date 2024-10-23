import pytest
import numpy as np
from evaluate import evaluate_predictions, evaluate_naive_baseline

def test_evaluate_predictions():
    x = np.array([[7, 8, 9], [10, 11, 12]])
    y = np.array([[1, 2, 3], [4, 5, 6]])
    y_pred = np.array([[1, 2, 3], [4, 5, 6]])

    # Test case for char tokenization
    results_char = evaluate_predictions(y_pred, x, y, "char")
    assert results_char["actual_test_accuracy"] == 1.0
    assert results_char["table_name_test_accuracy"] == 1.0
    assert results_char["pageid_test_accuracy"] == 1.0
    assert results_char["padding_test_accuracy"] == 1.0

    x = np.array([[7, 8, 9], [10, 11, 12]])
    y = np.array([[1, 2], [4, 5]])
    y_pred = np.array([[1, 2], [4, 5]])

    # Test case for word tokenization
    results_word = evaluate_predictions(y_pred, x, y, "word")
    assert results_word["actual_test_accuracy"] == 1.0
    assert results_word["table_name_test_accuracy"] == 1.0
    assert results_word["pageid_test_accuracy"] == 1.0
    assert results_word["padding_test_accuracy"] is None

    # Test case for invalid tokenization type
    with pytest.raises(ValueError, match="Invalid tokenization type: invalid"):
        evaluate_predictions(y_pred, x, y, "invalid")

def test_actual_test_accuracy():
    y_pred = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    y = np.array([[1, 2, 3], [4, 5, 7], [10, 11, 12]])
    x = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])  # Dummy input data

    results = evaluate_predictions(y_pred, x, y, "char")
    assert results["actual_test_accuracy"] == 1/3

    y_pred = np.array([[1, 2], [4, 5], [7, 8]])
    y = np.array([[1, 2], [4, 5], [10, 11]])
    x = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])  # Dummy input data

    results = evaluate_predictions(y_pred, x, y, "word")
    assert results["actual_test_accuracy"] == 2/3

def test_table_name_accuracy():
    y_pred = np.array([[1, 2, 3], [7, 5, 6], [7, 8, 9]])
    y = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    x = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])  # Dummy input data

    results = evaluate_predictions(y_pred, x, y, "char")
    assert results["table_name_test_accuracy"] == 2/3

    y_pred = np.array([[1, 2], [4, 5], [7, 8]])
    y = np.array([[1, 2], [4, 5], [10, 11]])
    x = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])  # Dummy input data

    results = evaluate_predictions(y_pred, x, y, "word")
    assert results["table_name_test_accuracy"] == 2/3

def test_pageid_accuracy():
    y_pred = np.array([[1, 2, 3, 0, 0], [4, 5, 6, 0, 0], [7, 8, 9, 0, 0]])
    y = np.array([[1, 2, 3, 0, 0], [4, 5, 6, 0, 0], [7, 8, 9, 0, 0]])
    x = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])  # Dummy input data

    results = evaluate_predictions(y_pred, x, y, "char")
    assert results["pageid_test_accuracy"] == 1.0

    y_pred = np.array([[-1, 2, 3, 0, 0], [14, 5, 6, 0, 0], [27, 8, 9, 0, 0]])
    y = np.array([[1, 2, 3, 0, 0], [4, 5, 6, 0, 0], [7, 8, 9, 0, 0]])
    x = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])  # Dummy input data

    results = evaluate_predictions(y_pred, x, y, "char")
    assert results["pageid_test_accuracy"] == 1.0

    y_pred = np.array([[-1, 2, 3, 0, 0], [14, 7, 6, 0, 0], [27, 8, 9, 0, 0]])
    y = np.array([[1, 2, 3, 0, 0], [4, 5, 6, 0, 0], [7, 8, 9, 0, 0]])
    x = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])  # Dummy input data

    results = evaluate_predictions(y_pred, x, y, "char")
    assert results["pageid_test_accuracy"] == 2/3

    y_pred = np.array([[1, 2], [4, 5], [7, 8]])
    y = np.array([[1, 2], [4, 5], [7, 8]])
    x = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])  # Dummy input data

    results = evaluate_predictions(y_pred, x, y, "word")
    assert results["pageid_test_accuracy"] == 1.0

    y_pred = np.array([[100, 2], [4, 5], [7, 8]])
    y = np.array([[1, 2], [4, 5], [7, 8]])
    x = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])  # Dummy input data

    results = evaluate_predictions(y_pred, x, y, "word")
    assert results["pageid_test_accuracy"] == 1.0

    y_pred = np.array([[100, 2], [4, 5000], [7, 8]])
    y = np.array([[1, 2], [4, 5], [7, 8]])
    x = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])  # Dummy input data

    results = evaluate_predictions(y_pred, x, y, "word")
    assert results["pageid_test_accuracy"] == 2/3

def test_evaluate_naive_baseline():
    y_test = np.array([[1, 1], [1, 1], [4, 5], [6, 7], [8, 9]])

    results = evaluate_naive_baseline(y_test)
    assert results["actual_test_accuracy"] == 1/4

    y_test = np.array([[1, 1], [1, 1], [4, 5], [4, 5], [8, 9]])

    results = evaluate_naive_baseline(y_test)
    assert results["actual_test_accuracy"] == 2/4

    y_test = np.array([[1, 1], [1, 1], [1, 1], [1, 1], [1, 1]])
    results = evaluate_naive_baseline(y_test)
    assert results["actual_test_accuracy"] == 1.0

    y_test = np.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
    results = evaluate_naive_baseline(y_test)
    assert results["actual_test_accuracy"] == 0.0

    # Test flatten procedure
    y_test = np.array([[[1, 1], [1, 1]], [[1, 1], [1, 1]], [[4, 4], [5, 5]], [[4, 4], [5, 5]], [[8, 8], [9, 9]]])

    results = evaluate_naive_baseline(y_test)
    assert results["actual_test_accuracy"] == 2/4


# TODO: Add more tests for pageid_test_accuracy and padding_test_accuracy

if __name__ == "__main__":
    pytest.main(["-v", __file__])