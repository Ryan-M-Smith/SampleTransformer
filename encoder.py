#
# FILENAME: encoder.py | Sample Transformer
# DESCRIPTION: The transformer encoder module
# CREATED: 2024-01-03 @ 11:14 PM
# COPYRIGHT: Copyright (c) 2023-2024 by Ryan Smith <rysmith2113@gmail.com>
#

import tensorflow as tf

from keras.layers import Dropout
from keras.initializers import GlorotUniform

from defaults import DROPOUT_RATE, EPSILON
from encoder_layer import EncoderLayer

class Encoder(tf.Module):
    def __init__(self, dim_model: int, dim_ffn: int, num_layers: int, num_heads: int,
                 dropout_rate: float, epsilon: float, name: str | None = None):
        super().__init__(name)
        
        # Create a list containing n_layers Encoder Layers 
        self.layers = [EncoderLayer(dim_model, dim_ffn, num_heads, dropout_rate, epsilon, dtype=tf.dtypes.float64) for _ in range(num_layers)]

        # Dropout layer
        self.dropout_layer = Dropout(dropout_rate)
    
    def __call__(self, tensor: tf.Tensor, tensor_mask: tf.Tensor = None) -> tf.Tensor:
        # Iterate over every Encoder Layer and pass the input tensor and the mask through
        # the layer. The output tensor will be used as the input tensor for the next passthrough.
        for layer in self.layers:
            tensor = layer(tensor, tensor_mask)
        
        return tensor