import pandas as pd
from keras.src.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical

ROW_CSV_FILE = r"row_locks.csv"

# Load and preprocess data
data = pd.read_csv(ROW_CSV_FILE)

# Strip spaces from column headers
data.columns = data.columns.str.strip()

# Create features
# TODO: add row_id, add token for rowid and page id token
# TODO: try transformer
data["page_table_combined"] = (
    # TODO: uncomment the line below  and comment the next one for char-based tokenization
    # data["PAGEID"].astype(str).apply(lambda x: " ".join(x))
    data["PAGEID"].astype(str)
    + "_"
    + data["TABNAME"].astype(str)
)


# Prepare sequences for LSTM
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


seq_length = 50  # Define sequence length
source_texts, target_texts = create_sequences(data, seq_length)

# Parameters
vocab_size = 10000  # Vocabulary size
embedding_dim = 128  # Embedding dimension
max_length = seq_length  # Maximum length of the input sequences
lstm_units = 64  # Number of LSTM units

# Tokenization
# TODO: create one unified tokenizer for input and output
source_tokenizer = Tokenizer(num_words=vocab_size)
source_tokenizer.fit_on_texts(source_texts)
source_sequences = source_tokenizer.texts_to_sequences(source_texts)
padded_source_sequences = pad_sequences(
    source_sequences, maxlen=max_length, padding="post"
)

target_tokenizer = Tokenizer(num_words=vocab_size)
target_tokenizer.fit_on_texts(target_texts)
target_sequences = target_tokenizer.texts_to_sequences(target_texts)
padded_target_sequences = pad_sequences(
    target_sequences, maxlen=max_length, padding="post"
)

# Shifting target sequences to be the expected output (next token)
input_data = padded_source_sequences
output_data = to_categorical(padded_target_sequences, num_classes=vocab_size)

# Partitioning data into train and test sets (70% train, 30% test)
split_index = int(len(input_data) * 0.7)
x_train, x_test = input_data[:split_index], input_data[split_index:]
y_train, y_test = output_data[:split_index], output_data[split_index:]

# Define the model
model = Sequential(
    [
        Embedding(
            input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length
        ),
        LSTM(lstm_units, return_sequences=True),
        Dense(vocab_size, activation="softmax"),
    ]
)

# Compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Display model summary
model.summary()

# Define the EarlyStopping callback
early_stopping = EarlyStopping(
    monitor="val_accuracy", patience=5, restore_best_weights=True
)

# Train the model
history = model.fit(
    x_train,
    y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping],
)

# Evaluate the model on the test data
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
