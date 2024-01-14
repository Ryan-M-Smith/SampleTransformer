import tensorflow as tf
from keras import Model
from keras.layers import Dense, Input, Lambda

from embedding import Embedding
from positional_encoding import PositionalEncoder
from attention import AttentionBlock
from encoder_layer import EncoderLayer
from decoder_layer import DecoderLayer

from defaults import *

# Example usage:
batch_size = 64
vocab_size = 1000
d_model = 256
d_ffn = d_model * 4
num_heads = 8
seq_len = 20

# Encoder inputs
encoder_inputs = Input(batch_size=batch_size, shape=(seq_len,), dtype=tf.dtypes.int64, name="english")
encoder_embedding = Embedding(vocab_size, d_model, name="encoder_embedding")(encoder_inputs)
encoder_inputs_with_pos = PositionalEncoder(d_model, DROPOUT_RATE, EXP_BASE, seq_len, name="positional_encoder1")(encoder_embedding)
encoder_outputs = EncoderLayer(d_model, d_ffn, num_heads, DROPOUT_RATE, EPSILON, dtype=tf.float64, name="encoder_layer")(encoder_inputs_with_pos)

# Decoder inputs
decoder_inputs = Input(batch_size=batch_size, shape=(seq_len,), dtype=tf.dtypes.int64, name="spanish")
decoder_embedding = Embedding(vocab_size, d_model, name="decoder_embedding")(decoder_inputs)
decoder_inputs_with_pos = PositionalEncoder(d_model, DROPOUT_RATE, EXP_BASE, seq_len, name="positional_encoder2")(decoder_embedding)
decoder_outputs, weights = DecoderLayer(d_model, d_ffn, vocab_size, num_heads, DROPOUT_RATE, EPSILON, dtype=tf.dtypes.float64, name="decoder_layer")(decoder_inputs_with_pos, encoder_outputs)

logits = Dense(vocab_size, name="logits")(decoder_outputs)
target_one_hot = Lambda(lambda x: tf.one_hot(x, vocab_size, dtype=tf.dtypes.float64), name="one_hot")(decoder_inputs)

# encoder_inputs = Input(batch_size=64, shape=(seq_len,), dtype=tf.dtypes.int64, name="english")

# encoder_inputs = tf.cast(Embedding(vocab_size, d_model)(encoder_inputs), dtype=tf.dtypes.float64)
# encoder_inputs = PositionalEncoder(d_model, DROPOUT_RATE, EXP_BASE, seq_len, name="positional_encoder1")(encoder_inputs)

# decoder_inputs = Input(batch_size=64, shape=(seq_len,), dtype=tf.dtypes.int64, name="spanish")

# decoder_inputs = tf.cast(Embedding(vocab_size, d_model)(decoder_inputs), dtype=tf.dtypes.float64)
# decoder_inputs = PositionalEncoder(d_model, DROPOUT_RATE, EXP_BASE, seq_len, name="positional_encoder2")(decoder_inputs)

# encoder_outputs = EncoderLayer(d_model, d_ffn, num_heads, DROPOUT_RATE, EPSILON, dtype=tf.dtypes.float64, name="encoder_layer")(encoder_inputs)
# decoder_outputs, weights = DecoderLayer(d_model, d_ffn, num_heads, DROPOUT_RATE, EPSILON, dtype=tf.dtypes.float64, name="decoder_layer")(decoder_inputs, encoder_outputs)

# Create a model
model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=[decoder_outputs, weights])

# Print model summary
model.summary()

print("\nOutputs:", decoder_outputs)
print("Weights:", weights)