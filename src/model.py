from keras.models import Model
from keras.layers import Embedding, LSTM, Dense, RepeatVector, Input
import keras_nlp

def build_transformer_model(
    vocab_size,
    max_length,
    out_seq_length,
    intermediate_dim=512,
    num_heads=8,
    dropout=0.1,
    hidden_dim=256,
    embedding_dim=128,
):
    input = Input(shape=(max_length,))
    embedding = keras_nlp.layers.TokenAndPositionEmbedding(
        vocab_size, max_length, embedding_dim
    )(input)
    transformer = keras_nlp.layers.TransformerEncoder(
        intermediate_dim=intermediate_dim, num_heads=num_heads, dropout=dropout
    )(embedding)[:, -out_seq_length:, :] # TODO: Replace slicing with keras ops
    dense = Dense(hidden_dim, activation="relu")(transformer)
    output = Dense(vocab_size, activation="softmax")(dense)

    model = Model(inputs=input, outputs=output)
    return model

def build_lstm_model(
    vocab_size,
    max_length,
    out_seq_length,
    lstm_units=256,
    embedding_dim=128,
    hidden_dim=256,
    position_embedding=False,
):
    input_layer = Input(shape=(max_length,))
    if position_embedding:
        embedding = keras_nlp.layers.TokenAndPositionEmbedding(
            vocab_size, max_length, embedding_dim
        )(input_layer)
    else:
        embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(input_layer)
    x = LSTM(lstm_units)(embedding)
    repeat = RepeatVector(out_seq_length)(x)
    dense = Dense(hidden_dim, activation="relu")(repeat)
    output_layer = Dense(vocab_size, activation="softmax")(dense)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model

