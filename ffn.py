#
# FILENAME: ffn.py | Sample Transformer
# DESCRIPTION: A Position-wise Feed-forward Network (FFN)
# CREATED: 2024-01-03 @ 4:49 PM
# COPYRIGHT: Copyright (c) 2023-2024 by Ryan Smith <rysmith2113@gmail.com>
#

import tensorflow as tf
from keras.layers import Dense, Dropout, Layer

from defaults import DROPOUT_RATE

class FeedForwardNetwork(Layer):
    def __init__(self, dim_model: int, dim_ffn: int, dropout_rate: int, trainable: bool = True, name: str | None = None,
                 dtype: tf.dtypes.DType | None = None, dynamic: bool = False, **kwargs) -> None:
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        
        self.dim_model = dim_model
        self.dim_ffn   = dim_ffn    # The size of the FFN. This is often set to 4 * d_model.
        
        #
        # The two linear layers of the Multi-layer Perceptron
        #
        # The first layer has shape (d_model, d_ffn) and applies ReLU activation. The second layer has shape
        # (d_ffn, d_model). When a tensor is passed through layer 1 and then layer 2, it gets reshaped, then
        # returned to it's original shape.
        #
        self.layer1 = Dense(units=dim_ffn, input_shape=(dim_model,), activation="relu", dtype=tf.dtypes.float64)
        self.layer2 = Dense(units=dim_model, input_shape=(dim_ffn,), dtype=tf.dtypes.float64)
        
        # Dropout layer
        self.dropout_layer = Dropout(dropout_rate, dtype=tf.dtypes.float64)
    
    def __call__(self, attention_probs: tf.Tensor) -> tf.Tensor:
        #
        # Pass the attention probabilities through the first layer. The input tensor has shape (batch_size, seq_len, d_model),
        # and when passed through a dense layer of shape (d_model, d_ffn), the resulting tensor will have shape
        # (batch_size, seq_len, d_ffn). ReLU activation is applied to the tensor in this layer, so all positive values stay
        # the same while values less than or equal to zero become zero.
        #
        # The resulting tensor is then passed through the dropout layer to get the final result. 
        #
        layer1_result = self.dropout_layer(self.layer1(attention_probs))
        
        #
        # Pass the layer 1 result through the second layer. The input tensor has shape (batch_size, seq_len, d_ffn),
        # and when passed through a dense layer of shape (d_ffn, d_model), the resulting tensor will have shape
        # (batch_size, seq_len, d_model), which is the same shape as the original attention probabilities tensor.
        #
        # The values in the resulting tensor are adjusted according to the weights and activation function.
        #
        layer2_result = self.layer2(layer1_result)
        
        return layer2_result
        