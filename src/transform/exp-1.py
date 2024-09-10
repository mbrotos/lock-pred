import pandas as pd


def clean_data(df):
    # Drop column: 'folder_path'
    df = df.drop(columns=["folder_path"])
    # Performed 5 aggregations grouped on columns: 'model', 'data' and 15 other columns
    df = (
        df.groupby(
            [
                "model",
                "data",
                "epochs",
                "batch_size",
                "learning_rate",
                "seq_length",
                "test_split",
                "val_split",
                "vocab_size",
                "tokenization",
                "patience",
                "results_dir",
                "experiment_name",
                "shuffle",
                "add_start_end_tokens",
                "add_row_id",
                "add_label_tokens",
            ]
        )
        .agg(
            actual_test_accuracy_mean=("actual_test_accuracy", "mean"),
            actual_test_accuracy_std=("actual_test_accuracy", "std"),
            loss_mean=("loss", "mean"),
            loss_std=("loss", "std"),
            accuracy_per_output_mean=("accuracy_per_output", "mean"),
            accuracy_per_output_std=("accuracy_per_output", "std"),
        )
        .reset_index()
    )
    return df


# Loaded variable 'df' from URI: /root/github/lock-pred/results/exp-1/results.csv
df = pd.read_csv("results/exp-1/results.csv")

df_clean = clean_data(df.copy())
df_clean.head()

df_clean.to_csv("results/exp-1/results-clean.csv", index=False)
