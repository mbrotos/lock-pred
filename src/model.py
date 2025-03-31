from keras.models import Model
from keras.layers import Embedding, LSTM, Dense, RepeatVector, Input, TimeDistributed
import keras_nlp
import keras
import numpy as np

def build_transformer_model_classifier(
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
    # Squeeze the output to remove the extra dimension if out_seq_length is 1
    output = keras.ops.squeeze(output, axis=1) if out_seq_length == 1 else output

    model = Model(inputs=input, outputs=output)
    return model

def build_transformer_model_regression(
    feature_dim,
    max_length,
    horizon,
    intermediate_dim=512,
    num_heads=8,
    dropout=0.1,
    hidden_dim=256,
    embedding_dim=128,
):
    input = Input(shape=(max_length, feature_dim))
    embedding = keras_nlp.layers.TokenAndPositionEmbedding(
        feature_dim, max_length, embedding_dim
    )(input)
    transformer = keras_nlp.layers.TransformerEncoder(
        intermediate_dim=intermediate_dim, num_heads=num_heads, dropout=dropout
    )(embedding)
    pooling = keras.layers.GlobalAveragePooling2D(data_format='channels_last')(transformer)
    dense = Dense(hidden_dim, activation="relu")(pooling)
    output = Dense(horizon, activation="sigmoid")(dense)

    model = Model(inputs=input, outputs=output)
    return model

def build_transformer_model_casual(
    vocab_size,
    max_length,
    intermediate_dim=512,
    num_heads=8,
    dropout=0.3,
    embedding_dim=128,
    num_layers=4
):
    input = Input(shape=(max_length,))
    embedding = keras_nlp.layers.TokenAndPositionEmbedding(
        vocab_size, max_length, embedding_dim
    )
    x = embedding(input)
    for _ in range(num_layers):
        decoder_layer = keras_nlp.layers.TransformerDecoder(
            intermediate_dim=intermediate_dim, num_heads=num_heads, dropout=dropout
        )
        x = decoder_layer(x)
    # Round vocab size to nearest power of 2
    vocab_size = 2 ** int(np.ceil(np.log2(vocab_size)))
    output = Dense(vocab_size, activation=None)(x) # No activation on the output

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
    x = LSTM(lstm_units, return_sequences=True)(repeat)
    output_layer = TimeDistributed(Dense(vocab_size, activation="softmax"))(x)
    # Squeeze the output to remove the extra dimension if out_seq_length is 1
    output_layer = keras.ops.squeeze(output_layer, axis=1) if out_seq_length == 1 else output_layer

    model = Model(inputs=input_layer, outputs=output_layer)
    return model

def build_lstm_model_regression(
    feature_dim,
    max_length,
    horizon,
    lstm_units=256,
    hidden_dim=256,
    dropout=0.2
):
    input_layer = keras.Input(shape=(max_length, feature_dim))
    
    # LSTM layer
    x = LSTM(lstm_units, return_sequences=False, dropout=dropout)(input_layer)
    
    # Dense layer(s)
    x = Dense(hidden_dim, activation='relu')(x)
    
    # Final output for regression
    output_layer = Dense(horizon, activation='sigmoid')(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model

def build_ann_model_regression(
    feature_dim,
    max_length,
    horizon,
    hidden_dim=256,
    dropout=0.2
):
    input_layer = keras.Input(shape=(max_length, feature_dim))
    
    # Dense layer(s)
    x = Dense(hidden_dim, activation='relu')(input_layer)
    x = keras.layers.Dropout(dropout)(x)
    x = Dense(hidden_dim, activation='relu')(x)
    x = keras.layers.Dropout(dropout)(x)
    x = Dense(hidden_dim, activation='relu')(x)
    x = keras.layers.Dropout(dropout)(x)
    x = Dense(hidden_dim, activation='relu')(x)
    x = keras.layers.Dropout(dropout)(x)
    
    # Final output for regression
    output_layer = Dense(horizon, activation='sigmoid')(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model
