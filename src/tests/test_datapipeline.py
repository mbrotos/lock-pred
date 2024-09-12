import pandas as pd
import numpy as np
import pytest
from datapipeline import (
    load_data,
    create_sequences,
    tokenize_data,
    split_data,
    prepare_datasets,
)


def test_load_data():
    # Create a sample DataFrame
    data = pd.DataFrame(
        {
            "PAGEID": ["123", "456", "789"],
            "ROWID": ["1", "2", "13"],
            "TABNAME": ["TABLE_1", "TABLE_2", "TABLE_3"],
            "TABSCHEMA": ["SCHEMA1", "SYSIBM", "SCHEMA3"],
        }
    )

    # Test default parameters
    result = load_data(data.copy())
    assert "input" in result.columns
    assert "output" in result.columns
    assert np.all(result["input"] == result["output"])
    assert result["output"].iloc[0] == "1 2 3 TABLE1"
    assert result["output"].iloc[1] == "4 5 6 TABLE2"

    # Test with char_based=False
    result = load_data(data.copy(), char_based=False)
    assert np.all(result["input"] == result["output"])
    assert result["output"].iloc[0] == "123 TABLE1"
    assert result["output"].iloc[1] == "456 TABLE2"

    # Test with add_row_id=True
    result = load_data(data.copy(), add_row_id=True)
    assert result["input"].iloc[0] == "1 1 2 3 TABLE1"
    assert result["output"].iloc[0] == "1 2 3 TABLE1"
    assert result["input"].iloc[1] == "2 4 5 6 TABLE2"
    assert result["output"].iloc[1] == "4 5 6 TABLE2"

    # Test with add_label_tokens=True
    result = load_data(data.copy(), add_label_tokens=True)
    assert result["input"].iloc[0] == "<PAGEID> 1 2 3 TABLE1"
    assert result["output"].iloc[0] == "1 2 3 TABLE1"
    assert result["input"].iloc[1] == "<PAGEID> 4 5 6 TABLE2"
    assert result["output"].iloc[1] == "4 5 6 TABLE2"

    # Test with add_start_end_tokens=True
    result = load_data(data.copy(), add_start_end_tokens=True)
    assert result["input"].iloc[0] == "<START> 1 2 3 TABLE1 <END>"
    assert result["output"].iloc[0] == "1 2 3 TABLE1"
    assert result["input"].iloc[1] == "<START> 4 5 6 TABLE2 <END>"
    assert result["output"].iloc[1] == "4 5 6 TABLE2"

    # Test with remove_system_tables=True
    result = load_data(data.copy(), remove_system_tables=True)
    assert len(result[result["TABSCHEMA"] == "SYSIBM"]) == 0
    assert np.all(result["input"] == result["output"])
    assert result["output"].iloc[0] == "1 2 3 TABLE1"
    assert result["output"].iloc[1] == "7 8 9 TABLE3"

    # Test with add_row_id=True and add_label_tokens=True
    result = load_data(data.copy(), add_row_id=True, add_label_tokens=True)
    assert result["input"].iloc[0] == "<ROWID> 1 <PAGEID> 1 2 3 TABLE1"
    assert result["output"].iloc[0] == "1 2 3 TABLE1"
    assert result["input"].iloc[1] == "<ROWID> 2 <PAGEID> 4 5 6 TABLE2"
    assert result["output"].iloc[1] == "4 5 6 TABLE2"

    # Test with add_row_id=True and add_label_tokens=True and char_based=False
    result = load_data(data.copy(), add_row_id=True, add_label_tokens=True, char_based=False)
    assert result["input"].iloc[0] == "<ROWID> 1 <PAGEID> 123 TABLE1"
    assert result["output"].iloc[0] == "123 TABLE1"
    assert result["input"].iloc[1] == "<ROWID> 2 <PAGEID> 456 TABLE2"
    assert result["output"].iloc[1] == "456 TABLE2"

    # Test with add_row_id=True and add_label_tokens=True and start_end_tokens=True
    result = load_data(data.copy(), add_row_id=True, add_label_tokens=True, add_start_end_tokens=True)
    assert result["input"].iloc[0] == "<START> <ROWID> 1 <PAGEID> 1 2 3 TABLE1 <END>"
    assert result["output"].iloc[0] == "1 2 3 TABLE1"
    assert result["input"].iloc[1] == "<START> <ROWID> 2 <PAGEID> 4 5 6 TABLE2 <END>"
    assert result["output"].iloc[1] == "4 5 6 TABLE2"
    assert result["input"].iloc[2] == "<START> <ROWID> 1 3 <PAGEID> 7 8 9 TABLE3 <END>"
    assert result["output"].iloc[2] == "7 8 9 TABLE3"

    # Test with add_row_id=True and add_label_tokens=True and start_end_tokens=True and char_based=False
    result = load_data(data.copy(), add_row_id=True, add_label_tokens=True, add_start_end_tokens=True, char_based=False)
    assert result["input"].iloc[0] == "<START> <ROWID> 1 <PAGEID> 123 TABLE1 <END>"
    assert result["output"].iloc[0] == "123 TABLE1"
    assert result["input"].iloc[1] == "<START> <ROWID> 2 <PAGEID> 456 TABLE2 <END>"
    assert result["output"].iloc[1] == "456 TABLE2"
    assert result["input"].iloc[2] == "<START> <ROWID> 13 <PAGEID> 789 TABLE3 <END>"
    assert result["output"].iloc[2] == "789 TABLE3"

    # TODO: Add more tests for different combinations of parameters

def test_create_sequences():
    data = pd.DataFrame(
        {"input": ["A B C", "D E F", "G H I", "J K L"], "output": ["X", "Y", "Z", "W"]}
    )

    X, y = create_sequences(data, seq_length=2)
    assert X == ["A B C D E F", "D E F G H I"]
    assert y == ["Z", "W"]

    # TODO: Add failure cases for mismatching sequence length


def test_tokenize_data():
    text = ["<START> hello world <END>", "hello python", "world of coding"]
    vocab_size = 10
    max_length = 5

    padded_sequences, tokenizer = tokenize_data(text, vocab_size, max_length)

    assert padded_sequences.shape == (3, 5)
    assert len(tokenizer.word_index) <= vocab_size
    # check that start and end tokens are in the tokenizer
    assert "<start>" in tokenizer.word_index.keys()
    assert "<end>" in tokenizer.word_index.keys()
    # check that the padded sequences are correct
    assert np.all(padded_sequences[0][-1] == 0.0)
    assert np.all(padded_sequences[1][2:] == 0.0)
    assert np.all(padded_sequences[2][3:] == 0.0)

    # TODO: Add more test cases for different vocab sizes and max lengths


def test_split_data():
    input_data = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    output_data = np.array([[0, 1], [1, 0], [1, 1], [0, 0], [1, 1]])
    test_size = 0.2

    x_train, x_test, y_train, y_test = split_data(input_data, output_data, test_size)

    # Test case for default behavior (no shuffle)
    assert len(x_train) == 4
    assert len(x_test) == 1
    assert len(y_train) == 4
    assert len(y_test) == 1
    np.testing.assert_array_equal(x_train, np.array([[1, 2], [3, 4], [5, 6], [7, 8]]))
    np.testing.assert_array_equal(x_test, np.array([[9, 10]]))
    np.testing.assert_array_equal(y_train, np.array([[0, 1], [1, 0], [1, 1], [0, 0]]))
    np.testing.assert_array_equal(y_test, np.array([[1, 1]]))

    # Test case with shuffle=True
    x_train_shuffled, x_test_shuffled, y_train_shuffled, y_test_shuffled = split_data(input_data, output_data, test_size, shuffle=True)
    
    assert len(x_train_shuffled) == 4
    assert len(x_test_shuffled) == 1
    assert len(y_train_shuffled) == 4
    assert len(y_test_shuffled) == 1
    
    # Check that the shuffled data is different from the non-shuffled data
    assert not np.array_equal(x_train, x_train_shuffled) or not np.array_equal(y_train, y_train_shuffled)

    # Test case with different test_size
    x_train_half, x_test_half, y_train_half, y_test_half = split_data(input_data, output_data, test_size=0.5)
    
    assert len(x_train_half) == 2
    assert len(x_test_half) == 3
    assert len(y_train_half) == 2
    assert len(y_test_half) == 3

# TODO: Add test for prepare_datasets
# def test_prepare_datasets():

if __name__ == "__main__":
    pytest.main()
