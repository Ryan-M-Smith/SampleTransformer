#
# FILENAME: layer_normalization.py | Sample Transformer
# DESCRIPTION: Transformer layer normalization from scratch
# CREATED: 2024-01-03 @ 8:11 PM
# COPYRIGHT: Copyright (c) 2023-2024 by Ryan Smith <rysmith2113@gmail.com>
#

import tensorflow as tf
from keras.layers import Layer

from defaults import EPSILON

class LayerNormalization(Layer):
    def __init__(self, axes: int | tf.TensorShape = -1, epsilon: float = EPSILON, gamma_init: str = "ones",
                 beta_init: str = "zeros", trainable: bool = True, name: str | None = None,
                 dtype: tf.dtypes.DType | None = None, dynamic: bool = False, **kwargs) -> None:
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        
        self.axes       = axes
        self.epsilon    = epsilon
        self.gamma_init = gamma_init
        self.beta_init  = beta_init
    
    def build(self, input_shape: tf.TensorShape) -> None:
        #
        # Create gamma and beta, the learnable parameters. The gamma value is multiplied by the
        # normalized tensor, and the beta value is added to that product. Gamma is the scale tensor,
        # and beta is the offset tensor. Both tensors have the same shape as the resulting normalized
        # tensor, (batch_size, seq_len, d_model). By default, gamma is a ones tensor and beta is a zeros
        # tensor. 
        #
        self.scale  = self.add_weight("scale", shape=(input_shape[-1],), initializer=self.gamma_init)
        self.offset = self.add_weight("offset", shape=(input_shape[-1],), initializer=self.beta_init)
    
    def call(self, tensor: tf.Tensor) -> tf.Tensor:
        #
        # Find the mean and variance of the input tensor. For transformers, the last axis in the
        # input tensor is normalized. The input tensor is typically attention probabilities from
        # the attention block, and will have shape (batch_size, seq_len, d_model).
        #
        mean, variance = tf.nn.moments(tensor, axes=self.axes, keepdims=True)
        
        #
        # Apply the normalization formula:
        #
        # y = (x - μ) / sqrt(σ^2 + ε)
        # Where:
        #   x   := the input tensor
        #   μ   := the mean of the last D dimensions
        #   σ^2 := the variance of the last D dimensions
        #   ε   := a small value that helps when σ^2 is small (1e-5 by default)
        #
        normalized_tensor = (tensor - mean) / tf.sqrt(variance + self.epsilon)
        
        # Return the normalized tensor with scale and offset factored in (learnable parameters)
        return tf.multiply(normalized_tensor, self.scale) + self.offset
        