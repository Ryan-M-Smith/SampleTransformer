#
# FILENAME: transformer.py | Sample Transformer
# DESCRIPTION: The transformer model
# CREATED: 2024-01-05 @ 3;12 AM
# COPYRIGHT: Copyright (c) 2023-2024 by Ryan Smith <rysmith2113@gmail.com>
#

import tensorflow as tf

from keras import Model, Sequential
from keras.initializers import GlorotUniform

from decoder import Decoder
from embedding import Embedding
from encoder import Encoder
from inputproc import build_source_mask, build_target_mask
from positional_encoding import PositionalEncoder

class Transformer(Model):
    def __init__(self, encoder: Encoder, decoder: Decoder, source_embed: Embedding,
                 target_embed: Embedding, positional_encoder: PositionalEncoder, source_pad_idx: int, target_pad_idx: int) -> None:
        super().__init__()
        
        # Transformer layers
        self.encoder = encoder
        self.decoder = decoder
        
        # Embedders
        self.source_embed = source_embed
        self.target_embed = target_embed
        
        # Padding indices
        self.source_pad_idx = source_pad_idx
        self.target_pad_idx = target_pad_idx
        
        #self.initialize_weights()
    
    def initialize_weights(self) -> None:
        """ Iterate through the layers in the model and initialize the weights """
        
        initializer = GlorotUniform()
        for layer in self.layers:
            for param in layer.trainable_variables:
                if len(param.get_shape()) > 1:  # Check if the parameter is a weight matrix
                    param.assign(initializer(shape=param.get_shape()))
    
    @tf.function
    def call(self, inputs: tuple[tf.Tensor, tf.Tensor], training=None, mask=None) -> tf.Tensor:
        print("Calling Transformer")
        
        source, target = inputs
        
        print("Source:", source)
        print("Target:", target)
        
        source_mask = build_source_mask(source, self.source_pad_idx)
        target_mask = build_target_mask(target, self.target_pad_idx)
        
        # Feed source through the encoder layer
        source = self.encoder(self.source_embed(source), source_mask)
        
        # Get decoder output (logits)
        output = self.decoder(self.target_embed(target), source, target_mask, source_mask)
        
        return output
        
def make_model(source_vocab_size: int, target_vocab_size: int, dim_model: int,
               dim_ffn: int, num_layers: int, num_heads: int, dropout_rate: float, epsilon: float,
               exp_base: int, max_seq_len: int) -> Transformer:
    """ Create and return a transformer model from its hyperparameters. """
    
    # Transformer layers
    encoder = Encoder(dim_model, dim_ffn, num_layers, num_heads, dropout_rate, epsilon)
    decoder = Decoder(target_vocab_size, dim_model, dim_ffn, num_layers, num_heads, dropout_rate, epsilon)
    
    source_embed = Embedding(source_vocab_size, dim_model) # Source embedding tensor
    target_embed = Embedding(target_vocab_size, dim_model) # Target embedding tensor
    
    positional_encoder = PositionalEncoder(dim_model, dropout_rate, exp_base, max_seq_len)
    
    # Sequential layers to pass the embeddings through the positional encoder
    #source_seq_layer = Sequential(layers=[source_embed, positional_encoder])
    #target_seq_layer = Sequential(layers=[target_embed, positional_encoder])
    
    return Transformer(encoder, decoder, source_embed, target_embed, positional_encoder, 0, 0)

