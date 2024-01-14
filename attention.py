#
# FILENAME: attention.py | Sample Transformer
# DESCRIPTION: Multi-head attention for a transformer
# CREATED: 2023-12-30 @ 9:49 PM
# COPYRIGHT: Copyright (c) 2023-2024 by Ryan Smith <rysmith2113@gmail.com>
#

from math import sqrt, inf

import tensorflow as tf
from keras.layers import Dense, Dropout, Layer

from defaults import DROPOUT_RATE

class AttentionBlock(Layer):
    def __init__(self, dim_model, num_heads: int, dropout_rate: float = DROPOUT_RATE, trainable: bool | None = True,
                 name: str | None = None, dtype: tf.dtypes.DType | None = None, dynamic: bool | None = False, **kwargs) -> None:
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        
        assert dim_model % num_heads == 0, "dim_model must be disible by num_heads"
        
        self.dim_model  = dim_model              # The dimension of the vectors used in the transformer
        self.num_heads  = num_heads              # The number of attention blocks 
        self.d_key      = dim_model // num_heads # The dimension of a given vector within an input tensor 
        
        #
        # Define linear layers for the transformer. These can be implemented with a
        # Keras dense layer, which receives a linear activation function by default.
        #
        # There is one linear layer for each of the three attention inputs; queries (Q)
        # keys (K), values (V), and one for the output (O).
        #
        # Every weights tensor has shape (d_model, d_model).
        # 
        self.query_weights  = Dense(dim_model, dtype=tf.dtypes.float64, name="query_weights")
        self.key_weights    = Dense(dim_model, dtype=tf.dtypes.float64, name="key_weights")
        self.value_weights  = Dense(dim_model, dtype=tf.dtypes.float64, name="value_weights")
        self.output_weights = Dense(dim_model, dtype=tf.dtypes.float64, name="output_weights")
        
        # Dropout layer
        self.dropout_layer = Dropout(dropout_rate, dtype=tf.dtypes.float64, name="attn_dropout")
    
    #@tf.autograph.experimental.do_not_convert
    def call(self, values: tf.Tensor, keys: tf.Tensor, queries: tf.Tensor, mask: tf.Tensor | None = None) -> tf.Tensor:
        batch_size = tf.shape(queries)[0]
        
        #
        # Pass the input tensors through the linear layers. To generate the queries, keys, and values tensors.
        # Each output tensor will have shape (batch_size, seq_len, d_model).
        #
        queries = self.query_weights(queries)
        keys    = self.key_weights(keys)
        values  = self.value_weights(values)
        
        # Split all the resulting across n_heads 
        queries = self.__split_heads(queries, batch_size)
        keys    = self.__split_heads(keys, batch_size)
        values  = self.__split_heads(values, batch_size)
        
        attention, weights = self.__attention(queries, keys, values, mask) # Calculate attention
        attention = self.__concat_heads(attention, batch_size)             # Concatenate the previously split heads
        attention = self.output_weights(attention)                         # Pass the attention tensor through the output linear layer

        output = self.output_weights(attention)
        
        return output, weights
    
    def __attention(self, queries: tf.Tensor, keys: tf.Tensor, values: tf.Tensor, mask: tf.Tensor | None = None) -> tf.Tensor:
        """
            Calculate scaled dot-product attention. Given three tensors, queries (`Q`), keys (`K`), and values (`V`):
            
            `Attention = softmax((Q * K^T) / sqrt(d_key)) * V`
        """
        
        ### NOTE: All input tensors have shape (batch_size, n_heads, seq_len, d_key).
        
        #
        # Transpose the rightmost two dimensions of the keys tensor before multiplication. The
        # input shape is (batch_size, n_heads, seq_len, d_key), and the shape of the resulting
        # tensor is (batch_size, n_heads, d_key, seq_len).
        #
        # Find the scaled dot product of the queries and transposed keys tensors. The result of this
        # operation is a tensor that shows the strength of each token in a sequence compared to
        # every token in the sequence (including itself). The resulting tensor is scaled by 1 / d_key
        # to modify the output results of the softmax function. The shape of the resulting tensor
        # is (batch_size, n_heads, seq_len, seq_len).
        #
        scaled_dot_prod = tf.matmul(queries, keys, transpose_b=True) / sqrt(self.d_key)
        
        #
        # When working with a collection of sequences of different lengths, the tokenized sequences are
        # padded with a number larger than the highest mapped value in the vocabulary. To avoid using these
        # values in the attention probability distribution, they must be replaced with a very small number
        # (such as -inf) so they become insignificant in the softmax calculation.
        #
        if mask is not None:
            scaled_dot_prod += mask * -inf
        
        #
        # Apply the softmax function to the scaled dot product to get context for each token. The resulting
        # tensor is a probability distribution of shape (batch_size, n_heads, seq_len, seq_len).
        # 
        attention_weights = tf.nn.softmax(scaled_dot_prod, axis=3) # The softmax dimension is seq_len (axis 3)
        
        #
        # Return the product of the attention probabilities (with dropout applied) and the values input tensor.
        # The shape of the resulting tensor is (batch_size, n_heads, seq_len, d_key).
        #
        dropout_weights = self.dropout_layer(attention_weights)
        return tf.matmul(dropout_weights, values), attention_weights
    
    def __split_heads(self, tensor: tf.Tensor, batch_size: int):
        """
            Split an input tensor across `n_heads` subspaces to create alternate different representations
            of a sequence.
        """
        
        tensor = tf.reshape(tensor, (batch_size, -1, self.num_heads, self.d_key))
        return tf.transpose(tensor, perm=[0, 2, 1, 3])
        
        #
        # All of the input tensors have shape (batch_size, seq_len, d_model).
        # Because d_key = d_model / n_heads => d_model = n_heads * d_key, we can
        # reshape each tensor to (batch_size, seq_len, n_heads, d_key)
        #
        #tensor = tf.reshape(tensor, [batch_size, -1, self.num_heads, self.d_key])
        
        #
        # Transpose seq_len and n_heads. The shape of the resulting tensor is (batch_size, n_heads, seq_len, d_key).
        # The resulting tensor contains batch_size sequences of n_heads heads. Each head has shape (seq_len, d_key). 
        # Essentially, each sequence's tokens are split from a (seq_len, d_model) subspace to to an (n_heads, seq_len, d_key)
        # subspace.  
        #
        #return tf.transpose(tensor, [0, 2, 1, 3])
    
    def __concat_heads(self, tensor: tf.Tensor, batch_size: int):
        """ Combine various embeddings from the attention step into one embedding. """
        
        #
        # Transpose n_heads and seq_len. The shape of the resulting tensor is (batch_size, seq_len, n_heads, d_key).
        # The resulting tensor is the same shape as the input tensor for __split_heads. 
        #
        tensor = tf.transpose(tensor, [0, 2, 1, 3])
        
        #
        # Reshape the tensor to (batch_size, seq_len, d_model). This creates one large result embedding with the same
        # shape as the initial attention input tensors.
        #
        tensor = tf.reshape(tensor, [batch_size, -1, self.dim_model])
        
        return tensor
    