#
# FILENAME: positional_encoding.py | Sample Transformer
# DESCRIPTION: Posititional encoding for an embedding
# CREATED: 2024-01-02 @ 9:40 PM
# COPYRIGHT: Copyright (c) 2023-2024 by Ryan Smith <rysmith2113@gmail.com>
#

from math import ceil, floor, log

import tensorflow as tf
from keras.layers import Dropout, Layer

class PositionalEncoder(Layer):
    def __init__(self, dim_model: int, dropout_rate: float, exp_base: int, max_seq_len: int, trainable: bool = True,
                 name: str | None = None, dtype: tf.dtypes.DType | None = None, dynamic: bool = False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        
        self.dim_model      = dim_model
        self.d_model_over_2 = int(ceil(self.dim_model / 2)) # d_model / 2, adjusted for odd values of d_model
        self.exp_base       = exp_base
        self.max_seq_len    = max_seq_len
        
        # Dropout layer
        self.dropout_layer  = Dropout(dropout_rate, dtype=tf.dtypes.float64)

    def call(self, embedding: tf.Tensor) -> tf.Tensor:
        pe_matrix = self.__build_pe_matrix() # Build the matrix of PE values
        
        #
        # Add the embeddings to the PE matrix. The PE matrix contains positional encoding values
        # for sequences all the way up to the max sequence length, so it must be resized to the 
        # sequence length of the embedding tensor before the two can be added. Gradient is disabled
        # for the PE matrix before it is added to the embedding so it won't be calculated during
        # backpropagation.
        #
        encoding = tf.cast(embedding, dtype=tf.dtypes.float64) + tf.stop_gradient(pe_matrix[:, :tf.shape(embedding)[1]])

        return self.dropout_layer(encoding) # Return the positional encoding with dropout applied
    
    def __build_pe_matrix(self) -> tf.Tensor:
        """
            Calculate the values multiplied by each position value `k` to create the positional encoding.
            The tensor returned from this function contains all possible values of `e^(-2i * ln(n) / d_model)` 
            for `0 <= i <= d_model / 2`.
            
            Because the value of the equation only changes every `2i` values, this calcuation can be opitmized as
            scalar multiplication of `-ln(n) / d_model` on a tensor containing every possible value of `2i` on
            `0 <= i <= d_model / 2`.
            
            Note: When calculating d_model / 2 for odd values of d_model, a mathematical adjustment must be made to
            ensure all tensors are the correct size. In the following function, the code
            
                >>> int(ceil(self.dim_model / 2))
            
            is used in place of `d_model / 2`. This value is stored in `self.d_model_over_2` for convenience.
        """
        
        #
        # Create a matrix to hold the positional encodings. The matrix has size (d_model, k)
        # and is a zero matrix by default.
        #
        pe_matrix = tf.zeros([self.max_seq_len, self.dim_model], dtype=tf.dtypes.float64)
        
        #
        # Create the position values (`k`-values) tensor. The tensor has shape (MAX_SEQUENCE_LEN, 1)
        # and contains 1D tensors containing values in the range [0, MAX_SEQUENCE_LEN).
        #
        # In order to use this tensor to calulate a Hadamard (element-wise) product, it must be
        # broadcasted to size (MAX_SEQUENCE_LEN, d_model / 2).
        #
        position_vals = tf.range(0, self.max_seq_len, dtype=tf.dtypes.float64)[:, tf.newaxis]
        
        #
        # Create the tensor of all possible values of e^(-2i * ln(n) / d_model). The tensor has shape
        # (1, d_model / 2). A tensor of all 2i values is multiplied by -ln(n) / d_model, then every value
        # in the resulting tensor is exponentiated to base e.
        #
        # In order to use this tensor to calulate a Hadamard (element-wise) product, it must be
        # broadcasted to size (MAX_SEQUENCE_LEN, d_model / 2).
        #
        exp_vals = tf.math.exp(tf.range(0, self.dim_model, 2, dtype=tf.dtypes.float64) * (-log(self.exp_base) / self.dim_model))
        exp_vals = tf.broadcast_to(exp_vals, shape=[self.max_seq_len, self.d_model_over_2])
        
        #
        # Find k * e^(-2i * ln(n) / d_model). This value is passed into sin (for even PE columns) and cos
        # (for odd PE columns) to fill the positional encoding tensor.
        #
        PE_VALUES     = tf.multiply(position_vals, exp_vals)
        EVEN_COL_VALS = tf.cast(tf.sin(PE_VALUES), dtype=tf.dtypes.float64)
        ODD_COL_VALS  = tf.cast(tf.cos(PE_VALUES), dtype=tf.dtypes.float64)[:, :(self.dim_model // 2)]
        
        #
        # Assign the PE values to the matrix. All even columns receive the value sin(PE_VALUE) while all
        # odd columns receive the value cos(PE_VALUE).
        #
        even_indices = tf.convert_to_tensor([(x, y) for x in range(0, self.max_seq_len) for y in range(0, self.dim_model, 2)])
        odd_indices  = tf.convert_to_tensor([(x, y) for x in range(0, self.max_seq_len) for y in range(1, self.dim_model, 2)])
        
        #
        # By default, the tensors of ordered pairs generated have the following shapes:
        #   Even indices: (ceil(d_model / 2) * MAX_SEQUENCE_LEN, 2)
        #   Odd indices: (floor(d_model / 2) * MAX_SEQUENCE_LEN, 2)
        #
        # To perform an update on the tensor, the ordered pairs need to be grouped into rows by
        # x-value. For example, all ordered pairs with x-value 0 should be in a row, etc. This
        # can be done by introducing a third dimension and reshaping the tensors to:
        #   Even indices: (MAX_SEQUENCE_LEN, ceil(d_model / 2), 2)
        #   Odd indices: (MAX_SEQUENCE_LEN, floor(d_model / 2), 2)
        #
        even_indices = tf.reshape(even_indices, [self.max_seq_len, int(ceil(self.dim_model / 2)), 2])
        odd_indices  = tf.reshape(odd_indices, [self.max_seq_len, int(floor(self.dim_model / 2)), 2])
        
        pe_matrix = tf.tensor_scatter_nd_update(pe_matrix, indices=even_indices, updates=EVEN_COL_VALS) # Insert the values for the even rows
        pe_matrix = tf.tensor_scatter_nd_update(pe_matrix, indices=odd_indices, updates=ODD_COL_VALS)   # Insert the values for the odd rows
        
        pe_matrix = tf.expand_dims(pe_matrix, 0) # Add an additional dimension for broadcasting across sequences
        
        return pe_matrix