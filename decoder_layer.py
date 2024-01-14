#
# FILENAME: decoder_layer.py | Sample Transformer
# DESCRIPTION: The transformer decoder layer
# CREATED: 2024-01-04 @ 4:52 PM
# COPYRIGHT: Copyright (c) 2023-2024 by Ryan Smith <rysmith2113@gmail.com>
#

import tensorflow as tf

from keras import Sequential
from keras.layers import Dense, Dropout, Layer

import masking

from attention import AttentionBlock
from defaults import DROPOUT_RATE, EPSILON
from ffn import FeedForwardNetwork
from layer_normalization import LayerNormalization

class DecoderLayer(Layer):
    def __init__(self, dim_model: int, dim_ffn: int, vocab_size: int, num_heads: int, dropout_rate: float,
                 epsilon: float, trainable: bool | None = True, name: str | None = None,
                 dtype: tf.dtypes.DType | None = None, dynamic: bool | None = False, **kwargs) -> None:
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        
        self.dim_model  = dim_model
        self.dim_ffn    = dim_ffn
        self.num_heads  = num_heads
        self.vocab_size = vocab_size
        
        #
        # Decoder Layer submodules
        #
        # The transformer decoder layer is made up of:
        #   * A masked Multi-head Attention block
        #   * A Layer Normalization block
        #   * A Multi-head Attention block
        #   * A second Layer Normalization Block
        #   * A Position-wise Feed-forward Network
        #   * A third Layer Normalization Block
        #
        self.masked_multihead_attn  = AttentionBlock(dim_model, num_heads, dropout_rate)
        self.masked_attn_layer_norm = LayerNormalization(epsilon=epsilon, dtype=dtype)
        self.multihead_attn         = AttentionBlock(dim_model, num_heads, dropout_rate)
        self.attn_layer_norm        = LayerNormalization(epsilon=epsilon, dtype=dtype)
        self.position_wise_ffn      = FeedForwardNetwork(dim_model, dim_ffn, dropout_rate)
        self.ffn_layer_norm         = LayerNormalization(epsilon=epsilon, dtype=dtype)
        
        # A sequential layer to produce logits
        self.logits                 = self.logits = Sequential([
                                                        Dense(dim_model, activation="relu", dtype=tf.dtypes.float64),
                                                        Dense(vocab_size, activation="softmax", dtype=tf.dtypes.float64)
                                                    ])
        
        # Dropout layer
        self.dropout_layer          = Dropout(dropout_rate, dtype=dtype)
    
    def call(self, target: tf.Tensor, source: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        src_padding_mask = masking.padding_mask(source)
        tgt_padding_mask = masking.padding_mask(target)
        causal_mask      = masking.causal_mask(target)
        combined_mask    = tf.math.minimum(tgt_padding_mask, causal_mask)
        
        # Calculate masked attention from the target embeddings, with the target mask applied
        attention, weights = self.masked_multihead_attn(target, target, target, causal_mask)
        
        # Calculate dropout for the attention tensor, add it to the target tensor, and normalize the result
        normalized_attn = self.masked_attn_layer_norm(target + self.dropout_layer(attention))
        
        # Calculate attention from the normalized masked attention and the source embeddings, with the source mask applied
        attention, weights = self.multihead_attn(normalized_attn, source, source, combined_mask)
        
        # Calculate dropout for the attention tensor, add it to the nornalized attention tensor, and normalize the result
        normalized_attn = self.attn_layer_norm(normalized_attn + self.dropout_layer(attention))
        
        # Pass the normalized attention tensor through the FFN
        ffn_result = self.position_wise_ffn(normalized_attn)
        
        # Calculate dropout for the FFN result tensor, add it to the normalized attention tensor, and normalize the result
        normalized_ffn_result = self.ffn_layer_norm(normalized_attn + self.dropout_layer(ffn_result))
        
        # Pass the result through the output layer
        logits = self.logits(normalized_ffn_result)
        
        return logits, weights
        