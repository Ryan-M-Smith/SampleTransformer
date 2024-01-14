#
# FILENAME: embedding.py | Sample Transformer
# DESCRIPTION: Generate an embedding for an input
# CREATED: 2023-12-31 @ 8:06 PM
# COPYRIGHT: Copyright (c) 2023-2024 by Ryan Smith <rysmith2113@gmail.com>
#

from math import sqrt

import tensorflow as tf
from keras.layers import Embedding as KerasEmbedding, Layer

class Embedding(Layer):
    def __init__(self, vocab_size: int, dim_model: int, trainable: bool | None = True,
                 name: str | None = None, dtype: tf.dtypes.DType | None = None,
                 dynamic: bool | None = False, **kwargs) -> None:
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        
        self.vocab_size = vocab_size
        self.dim_model  = dim_model
        self.LUT        = KerasEmbedding(self.vocab_size, self.dim_model, dtype=dtype) # Create a tensor the base embedding weights
    
    def __call__(self, indices: tf.Tensor) -> tf.Tensor:
        indices = tf.cast(indices, dtype=tf.dtypes.int64)
        embeddings = self.LUT(indices)            # Create and return an embedding
        return embeddings * sqrt(self.dim_model)  # Return the embeddings tensor scaled by sqrt(d_model)
