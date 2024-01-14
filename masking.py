#
# FILENAME: masking.py | Sample Transformer
# DESCRIPTION: Functions to generate mask tensors
# CREATED: 2024-01-10 @ 4:46 PM
# COPYRIGHT: Copyright (c) 2023-2024 by Ryan Smith <rysmith2113@gmail.com>
#

import tensorflow as tf

def padding_mask(tensor: tf.Tensor, value: int = 0) -> tf.Tensor:
    """ A mask applied to input sequences to keep the attention layer from attending padding values during training. """

    #
    # Generate the padding mask tensor. At each value in the tensor, if the value doesn't equal the padding
    # value, the mask contains True, or 1. If the value does equal the padding value, the mask contains
    # False, or zero. The resulting mask has shape (batch_size, seq_len), which is the same as the input
    # tensor.
    #
    mask = tf.math.not_equal(tensor, value)
    mask = tf.reduce_all(mask, axis=-1)  # Check if all values along the last axis are non-padding
    mask = tf.cast(mask, dtype=tf.float64)
    mask = mask[:, tf.newaxis, tf.newaxis, :]
    
    return mask

def causal_mask(target: tf.Tensor) -> tf.Tensor:
    """ A mask applied to target values in the decoder to prevent positions from attending future positions. """

    shape = tf.shape(target, out_type=tf.dtypes.int32)
    batch_size, seq_len = shape[0], shape[1]

    # # Create a lower triangle ones tensor to be used as a submask
    # submask = tf.linalg.band_part(tf.ones((seq_len, seq_len), dtype=tf.dtypes.int64), -1, 0)
    
    # # Reshape the submask to have the same shape as the original mask
    # submask = tf.reshape(submask, (1, seq_len, seq_len))
    
    # # Create a multiplication tensor to create batch_size masks
    # mult = tf.concat(
    #     [
    #         tf.cast(tf.expand_dims(batch_size, -1), dtype=tf.dtypes.int64), 
    #         tf.constant([1, 1], dtype=tf.dtypes.int64)
    #     ],
    #     axis=0
    # )
    
    # # Construct the mask
    # target_mask = tf.cast(tf.tile(submask, mult), dtype=tf.dtypes.float64)
    
    # # Reshape the mask tensor to (batch_size, 1, seq_len, seq_len) to broadcast across a tensor of shape
    # # (batch_size, n_heads, seq_len, seq_len)
    # return target_mask[:, tf.newaxis, tf.newaxis, :]
    
    # Create a lower triangular matrix
    lower_triangle = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)

    # Expand dimensions to match the required shape (batch_size, 1, 1, seq_len)
    causal_mask = tf.expand_dims(tf.expand_dims(lower_triangle, 0), 0)

    # Repeat for each sequence in the batch
    causal_mask = tf.tile(causal_mask, [batch_size, 1, 1, 1])

    return tf.cast(causal_mask, dtype=tf.dtypes.float64)