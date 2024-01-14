#
# FILENAME: encoder_layer.py | Sample Transformer
# DESCRIPTION: The transformer encoder layer
# CREATED: 2024-01-03 @ 10:07 PM
# COPYRIGHT: Copyright (c) 2023-2024 by Ryan Smith <rysmith2113@gmail.com>
#

import tensorflow as tf
from keras.layers import Dropout, Layer

import masking

from attention import AttentionBlock
from defaults import DROPOUT_RATE, EPSILON
from ffn import FeedForwardNetwork
from layer_normalization import LayerNormalization

class EncoderLayer(Layer):
    def __init__(self, dim_model: int, dim_ffn: int, num_heads: int, dropout_rate: float,
                 epsilon: float, trainable: bool | None = True, name: str | None = None,
                 dtype: tf.dtypes.DType | None = None, dynamic: bool | None = False, **kwargs) -> None:
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
    
        self.dim_model = dim_model
        self.dim_ffn   = dim_ffn
        self.num_heads = num_heads
        
        #
        # Encoder Layer submodules
        #
        # The transformer encoder layer is made up of:
        #   * A Multi-head Attention block
        #   * A Layer Normalization block
        #   * A Position-wise Feed-forward Network
        #   * Another Layer Normalization Block
        #
        self.multihead_attn     = AttentionBlock(dim_model, num_heads, dropout_rate)
        self.attn_layer_norm    = LayerNormalization(epsilon=epsilon, dtype=dtype)
        self.position_wise_ffn  = FeedForwardNetwork(dim_model, dim_ffn, dropout_rate)
        self.ffn_layer_norm     = LayerNormalization(epsilon=epsilon,dtype=dtype)
        
        # Dropout layer
        self.dropout_layer      = Dropout(dropout_rate, dtype=dtype)
    
    def call(self, source: tf.Tensor) -> tf.Tensor:
        source_mask = masking.padding_mask(source)
        
        # Calculate masked attention from the embeddings
        attention, _ = self.multihead_attn(source, source, source, source_mask)
        
        # Calculate dropout for the attention tensor, add it to the embeddings tensor, and normalize the result
        normalized_attn = self.attn_layer_norm(source + self.dropout_layer(attention))
        
        # Pass the normalized attention tensor through the FFN
        ffn_result = self.position_wise_ffn(normalized_attn)
        
        # Calculate dropout for the FFN result tensor, add it to the normalized attention tensor, and normalize the result
        normalized_ffn = self.ffn_layer_norm(normalized_attn + self.dropout_layer(ffn_result))
        
        return normalized_ffn