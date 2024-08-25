import pandas as pd
from keras.src.callbacks import EarlyStopping
from keras.models import Model
from keras.layers import Embedding, LSTM, Dense, RepeatVector, Input, Bidirectional
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
import numpy as np

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
    + " "
    + data["TABNAME"].astype(str).apply(lambda x: x.replace("_", ""))
)


# Prepare sequences for LSTM
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        x_cur = (
            data.iloc[i : i + seq_length][["page_table_combined"]]
            .apply(" ".join)
            .reset_index()
            .values[0][1][:]
        )
        X.append(x_cur)
        y_cur = data.iloc[i + seq_length]["page_table_combined"].split()
        y.append(y_cur[0])
        x_next = " ".join(x_cur.split()[1:]) + " " + y_cur[0]
        X.append(x_next)
        y.append(y_cur[1])
    return X, y


seq_length = 25  # Define sequence length
out_seq_length = 1  # Define output sequence length I.e., page_id and table_name
source_texts, target_texts = create_sequences(data, seq_length)

# Parameters
vocab_size = 900  # Vocabulary size
embedding_dim = 128  # Embedding dimension
max_length = seq_length  # Maximum length of the input sequences
lstm_units = 256  # Number of LSTM units

def check_oov(tokenized_texts):
    """Check how many OOV tokens are present in the tokenized texts"""
    for text in tokenized_texts:
        if 1 in text:
            return True
    return False
    

# Tokenization
# TODO: create one unified tokenizer for input and output
source_tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
source_tokenizer.fit_on_texts(source_texts)
source_sequences = source_tokenizer.texts_to_sequences(source_texts)
padded_source_sequences = pad_sequences(
    source_sequences, maxlen=max_length, padding="post"
)
if check_oov(source_sequences):
    raise ValueError("OOV tokens found in source sequences")

target_tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
target_tokenizer.fit_on_texts(target_texts)
target_sequences = target_tokenizer.texts_to_sequences(target_texts)
padded_target_sequences = pad_sequences(
    target_sequences, maxlen=out_seq_length, padding="post"
)
if check_oov(target_sequences):
    raise ValueError("OOV tokens found in target sequences")

# Shifting target sequences to be the expected output (next token)
input_data = padded_source_sequences
output_data = to_categorical(padded_target_sequences, num_classes=vocab_size)

# Partitioning data into train and test sets (70% train, 30% test) randomly but maintaining the order
# The unshuffled dataset results in a highly skewed dataset split, and the model performs poorly on test
indices = np.arange(len(input_data))
indices = list(zip(indices[::2], indices[1::2]))
np.random.shuffle(indices)
# Calculate the split index
split_index = int(len(indices) * 0.7)

# Partition the indices into training and testing sets
train_indices = indices[:split_index]
test_indices = indices[split_index:]

# flatten the indices
train_indices = [item for sublist in train_indices for item in sublist]
test_indices = [item for sublist in test_indices for item in sublist]

# Use the shuffled indices to partition the data
x_train, x_test = input_data[train_indices], input_data[test_indices]
y_train, y_test = output_data[train_indices], output_data[test_indices]

# Define the model
input_layer = Input(shape=(max_length,))
embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(input_layer)
x = LSTM(lstm_units)(embedding)
dense = Dense(256, activation="relu")(x)
output_layer = Dense(vocab_size, activation="softmax")(dense)

model = Model(inputs=input_layer, outputs=output_layer)

# Compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Display model summary
model.summary()

# Define the EarlyStopping callback
early_stopping = EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True
)

# Train the model
history = model.fit(
    x_train,
    y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping],
   # add weights
)

# Evaluate the model on the test dataset
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")




# Evaluate the model on a single example from test
loss, accuracy = model.evaluate(x_test[np.newaxis,0], y_test[np.newaxis,0])
print(f"Input text:", source_tokenizer.sequences_to_texts([x_test[0]]))
print(f"Expected output text:", target_tokenizer.sequences_to_texts([[np.argmax(y_test[0], axis=-1)]])
)
print(f"Test Accuracy (n=1): {accuracy * 100:.2f}%")

loss, accuracy = model.evaluate(x_test[::2], y_test[::2])
print(f"Test First Token Accuracy: {accuracy * 100:.2f}%")

loss, accuracy = model.evaluate(x_test[1::2], y_test[1::2])
print(f"Test Second Token Accuracy: {accuracy * 100:.2f}%")

# Manually predict on a single example and print the outputs

print("Predicting on a five example")

for i in range(15):
    print("Input:", x_test[i])
    print("Input text:", source_tokenizer.sequences_to_texts([x_test[i]]))
    #print("Expected output:", y_test[i])
    print("Expected output text:", target_tokenizer.sequences_to_texts([[np.argmax(y_test[i], axis=-1)]]))
    output = model.predict(x_test[np.newaxis,i])
    #print("Predicted output softmax:", output)
    print("Predicted output:", np.argmax(output, axis=-1))
    print("Predicted output text:", target_tokenizer.sequences_to_texts([np.argmax(output, axis=-1)]))
