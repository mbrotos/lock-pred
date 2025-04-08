import pandas as pd
import numpy as np
import pytest
from datapipeline import (
    load_data,
    create_sequences,
    create_sequences_token,
    tokenize_data,
    split_data,
    prepare_datasets,
    load_table_lock_data
)


def test_load_data():
    # Create a sample DataFrame
    data = pd.DataFrame(
        {
            "PAGEID": ["123", "456", "789", "72033"],
            "ROWID": ["1", "2", "13", "40"],
            "TABNAME": ["TABLE_1", "TABLE_2", "TABLE_3", "ORDERLINE"],
            "TABSCHEMA": ["SCHEMA1", "SYSIBM", "SCHEMA3", "SCHEMA3"],
            "Start Timestamp": ["2024-12-17-13.28.04.000003000", "2024-12-17-13.28.04.000033000", "2024-12-17-13.29.07.000263000", "2024-12-17-13.29.07.000265000"],
            "End Timestamp": ["2024-12-17-13.28.04.000003300", "2024-12-17-13.28.04.000033300", "2024-12-17-13.29.07.000263200", "2024-12-17-13.29.07.000265700"],
        }
    )

    # Test default parameters
    result, _ = load_data(data.copy())
    assert "input" in result.columns
    assert "output" in result.columns
    assert np.all(result["input"] == result["output"])
    assert result["output"].iloc[0] == "TABLE1 1 2 3"
    assert result["output"].iloc[1] == "TABLE2 4 5 6"

    # Test with char_based=False
    result, _ = load_data(data.copy(), char_based=False)
    assert np.all(result["input"] == result["output"])
    assert result["output"].iloc[0] == "TABLE1 123"
    assert result["output"].iloc[1] == "TABLE2 456"

    # Test with add_row_id=True
    result, _ = load_data(data.copy(), add_row_id=True)
    assert result["input"].iloc[0] == "TABLE1 1 1 2 3"
    assert result["output"].iloc[0] == "TABLE1 1 2 3"
    assert result["input"].iloc[1] == "TABLE2 2 4 5 6"
    assert result["output"].iloc[1] == "TABLE2 4 5 6"

    # Test with add_label_tokens=True
    result, _ = load_data(data.copy(), add_label_tokens=True)
    # duplicate results rows in dataframe
    result = pd.concat([result, result])
    assert result["input"].iloc[0] == "TABLE1 <PAGEID> 1 2 3"
    assert result["output"].iloc[0] == "TABLE1 1 2 3"
    assert result["input"].iloc[1] == "TABLE2 <PAGEID> 4 5 6"
    assert result["output"].iloc[1] == "TABLE2 4 5 6"
    assert result["input"].iloc[2] == "TABLE3 <PAGEID> 7 8 9"
    assert result["output"].iloc[2] == "TABLE3 7 8 9"
    assert result["input"].iloc[3] == "ORDERLINE <PAGEID> 7 2 0 3 3"
    assert result["output"].iloc[3] == "ORDERLINE 7 2 0 3 3"

    # Test with add_start_end_tokens=True
    result, _ = load_data(data.copy(), add_start_end_tokens=True)
    assert result["input"].iloc[0] == "<START> TABLE1 1 2 3 <END>"
    assert result["output"].iloc[0] == "TABLE1 1 2 3"
    assert result["input"].iloc[1] == "<START> TABLE2 4 5 6 <END>"
    assert result["output"].iloc[1] == "TABLE2 4 5 6"

    # Test with remove_system_tables=True
    result, _ = load_data(data.copy(), remove_system_tables=True)
    assert len(result[result["TABSCHEMA"] == "SYSIBM"]) == 0
    assert np.all(result["input"] == result["output"])
    assert result["output"].iloc[0] == "TABLE1 1 2 3"
    assert result["output"].iloc[1] == "TABLE3 7 8 9"

    # Test with add_row_id=True and add_label_tokens=True
    result, _ = load_data(data.copy(), add_row_id=True, add_label_tokens=True)
    assert result["input"].iloc[0] == "TABLE1 <ROWID> 1 <PAGEID> 1 2 3"
    assert result["output"].iloc[0] == "TABLE1 1 2 3"
    assert result["input"].iloc[1] == "TABLE2 <ROWID> 2 <PAGEID> 4 5 6"
    assert result["output"].iloc[1] == "TABLE2 4 5 6"

    # Test with add_row_id=True and add_label_tokens=True and char_based=False
    result, _ = load_data(data.copy(), add_row_id=True, add_label_tokens=True, char_based=False)
    assert result["input"].iloc[0] == "TABLE1 <ROWID> 1 <PAGEID> 123"
    assert result["output"].iloc[0] == "TABLE1 123"
    assert result["input"].iloc[1] == "TABLE2 <ROWID> 2 <PAGEID> 456"
    assert result["output"].iloc[1] == "TABLE2 456"

    # Test with add_row_id=True and add_label_tokens=True and start_end_tokens=True
    result, _ = load_data(data.copy(), add_row_id=True, add_label_tokens=True, add_start_end_tokens=True)
    assert result["input"].iloc[0] == "<START> TABLE1 <ROWID> 1 <PAGEID> 1 2 3 <END>"
    assert result["output"].iloc[0] == "TABLE1 1 2 3"
    assert result["input"].iloc[1] == "<START> TABLE2 <ROWID> 2 <PAGEID> 4 5 6 <END>"
    assert result["output"].iloc[1] == "TABLE2 4 5 6"
    assert result["input"].iloc[2] == "<START> TABLE3 <ROWID> 1 3 <PAGEID> 7 8 9 <END>"
    assert result["output"].iloc[2] == "TABLE3 7 8 9"

    # Test with add_row_id=True and add_label_tokens=True and start_end_tokens=True and char_based=False
    result, _ = load_data(data.copy(), add_row_id=True, add_label_tokens=True, add_start_end_tokens=True, char_based=False)
    assert result["input"].iloc[0] == "<START> TABLE1 <ROWID> 1 <PAGEID> 123 <END>"
    assert result["output"].iloc[0] == "TABLE1 123"
    assert result["input"].iloc[1] == "<START> TABLE2 <ROWID> 2 <PAGEID> 456 <END>"
    assert result["output"].iloc[1] == "TABLE2 456"
    assert result["input"].iloc[2] == "<START> TABLE3 <ROWID> 13 <PAGEID> 789 <END>"
    assert result["output"].iloc[2] == "TABLE3 789"

    # TODO: Add more tests for different combinations of parameters

def test_load_table_lock_data():
    data = pd.DataFrame(
        {
            "TABNAME": ["TABLE_1", "TABLE_2  ", "TABLE_3  ", "ORDERLINE"],
            "TABSCHEMA": ["SCHEMA1", "SYSIBM  ", "SCHEMA3", "SCHEMA3"],
            "Start Timestamp": ["2024-12-17-13.28.04.000003000", "2024-12-17-13.28.04.000033000", "2024-12-17-13.29.07.000263000", "2024-12-17-13.29.07.000265000"],
            "End Timestamp": ["2024-12-17-13.28.04.000003300", "2024-12-17-13.28.04.000033300", "2024-12-17-13.29.07.000263200", "2024-12-17-13.29.07.000265700"],
        }
    )
    result = load_table_lock_data(data.copy())
    assert result["input"].iloc[0] == "TABLE1"
    assert result["output"].iloc[0] == "TABLE1"
    assert result["input"].iloc[1] == "TABLE2"
    assert result["output"].iloc[1] == "TABLE2"
    
    # Test with remove_system_tables=True
    result = load_table_lock_data(data.copy(), remove_system_tables=True)
    assert len(result[result["TABSCHEMA"] == "SYSIBM"]) == 0
    assert np.all(result["input"] == result["output"])
    assert result["input"].iloc[0] == "TABLE1"
    assert result["output"].iloc[0] == "TABLE1"
    assert result["input"].iloc[1] == "TABLE3"
    assert result["output"].iloc[1] == "TABLE3"

def test_create_sequences():
    data = pd.DataFrame(
        {"input": ["A B C", "D E F", "G H I", "J K L"], "output": ["X", "Y", "Z", "W"]}
    )

    X, y = create_sequences(data, seq_length=2)
    assert X == ["A B C D E F", "D E F G H I"]
    assert y == ["Z", "W"]

    # TODO: Add failure cases for mismatching sequence length

def test_create_sequences_token():
    data = pd.DataFrame(
        {
            "input": ["A B C", "D E F", "G H I", "J K L"],
            "output": ["X", "Y", "Z", "W"],
            "Start Unix Timestamp": [1734442084000003000, 1734442084000033000, 1734442147000263000, 1734442147000265000],
            "End Unix Timestamp": [1734442084000003300, 1734442084000033300, 1734442147000263200, 1734442147000265700],
        }
    )

    X, y, _, _, _ = create_sequences_token(data, token_length=6)
    assert X == ["A B C D E F", "D E F G H I"]
    assert y == ["Z", "W"]

    X, y, _, _, _ = create_sequences_token(data, token_length=4)
    assert X == ["A B C", "D E F", "G H I"]
    assert y == ['Y', 'Z', 'W']

    X, y, _, _, _ = create_sequences_token(data, token_length=3)
    assert X == ["A B C", "D E F", "G H I"]
    assert y == ['Y', 'Z', 'W']

    # increase horizon to 2

    X, y, _, _, _ = create_sequences_token(data, token_length=6, horizon=2)
    assert X == ["A B C D E F"]
    assert y == ["Z W"]

    X, y, _, _, _ = create_sequences_token(data, token_length=4, horizon=2)
    assert X == ["A B C", "D E F"]
    assert y == ['Y Z', 'Z W']

    X, y, _, _, _ = create_sequences_token(data, token_length=3, horizon=2)
    assert X == ["A B C", "D E F"]
    assert y == ['Y Z', 'Z W']

    data = pd.DataFrame(
        {
            "input": ["A B C", "D E F", "G H I", "J K L", "M N O"],
            "output": ["X", "Y", "Z", "W", "V"],
            "Start Unix Timestamp": [1734442084000003000, 1734442084000033000, 1734442147000263000, 1734442147000265000, 1734442147000266000],
            "End Unix Timestamp": [1734442084000003300, 1734442084000033300, 1734442147000263200, 1734442147000265700, 1734442147000266700],
        }
    )

    X, y, _, _, _ = create_sequences_token(data, token_length=6, horizon=2)
    assert X == ["A B C D E F", "D E F G H I"]
    assert y == ["Z W", "W V"]

    # TODO: Test the output time sequences which are currently being ignored

    # TODO: Test the naive outputs

    
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

    x_train, x_test, y_train, _, y_test, _, = split_data(input_data, output_data, [], test_size)

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
    x_train_shuffled, x_test_shuffled, y_train_shuffled, _, y_test_shuffled, _, = split_data(input_data, output_data, [], test_size, shuffle=True)
    
    assert len(x_train_shuffled) == 4
    assert len(x_test_shuffled) == 1
    assert len(y_train_shuffled) == 4
    assert len(y_test_shuffled) == 1
    
    # Check that the shuffled data is different from the non-shuffled data
    assert not np.array_equal(x_train, x_train_shuffled) or not np.array_equal(y_train, y_train_shuffled)

    # Test case with different test_size
    x_train_half, x_test_half, y_train_half, _, y_test_half, _, = split_data(input_data, output_data, [], test_size=0.5)
    
    assert len(x_train_half) == 2
    assert len(x_test_half) == 3
    assert len(y_train_half) == 2
    assert len(y_test_half) == 3

    # TODO: Test the naive outputs 

# TODO: Add test for prepare_datasets
# def test_prepare_datasets():

# TODO: Add test for timestamp conversion
# def test_timestamp_conversion():

if __name__ == "__main__":
    pytest.main(["-v", __file__])
