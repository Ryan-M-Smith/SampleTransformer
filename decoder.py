#
# FILENAME: decoder.py | Sample Transformer
# DESCRIPTION: The transformer decoder module
# CREATED: 2024-01-05 @ 2:05 AM
# COPYRIGHT: Copyright (c) 2023-2024 by Ryan Smith <rysmith2113@gmail.com>
#

from typing import Any
import tensorflow as tf

from keras.layers import Dense, Dropout

from decoder_layer import DecoderLayer
from defaults import DROPOUT_RATE, EPSILON

class Decoder(tf.Module):
    def __init__(self, vocab_size: int, dim_model: int, dim_ffn: int, num_layers: int, num_heads: int,
                 dropout_rate: float, epsilon: float, name: str | None = None) -> None:
        super().__init__(name)
        
        # Create a list containing n_layers decoder layers
        self.layers = [DecoderLayer(dim_model, dim_ffn, num_heads, dropout_rate, epsilon, dtype=tf.dtypes.float64) for _ in range(num_layers)]
        
        # Output weights, for calculating the logits
        self.output_weights = Dense(units=vocab_size, input_shape=(dim_model,), dtype=tf.dtypes.float64)
        
        # Dropout layer
        self.dropout_layer = Dropout(dropout_rate)
    
    def __call__(self, target: tf.Tensor, source: tf.Tensor, target_mask: tf.Tensor, source_mask: tf.Tensor) -> tf.Tensor:
        # Iterate over every Decoder Layer and pass the input tensors and the masks through
        # the layer. The output tensor will be used as an input tensor for the next passthrough.
        for layer in self.layers:
            target = layer(target, source, target_mask, source_mask)
        
        # Pass the target tensor through the output linear layer and return the result
        return self.output_weights(target)
        
        