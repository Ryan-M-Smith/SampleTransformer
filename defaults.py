#
# FILENAME: defaults.py | Sample Transformer
# DESCRIPTION: Various constant values for the transformer
# CREATED: 2024-01-03 @ 12:16 AM
# COPYRIGHT: Copyright (c) 2023-2024 by Ryan Smith <rysmith2113@gmail.com>
#

"""
    Various constant values for the transformer. Modifying the values of the these constants
    changes the default behavior of the transformer.
"""

#
# DROPOUT_RATE
# The rate at which the neural network drops units and their connections. Droput is applied to prevent co-adaption.
# Default value: 0.1
#
DROPOUT_RATE = 0.1

#
# EPOCHS
# The number of epochs to use during training
# Default value: 30
#
EPOCHS = 5

#
# EPSILON
# A small number used in Layer Normalization to help with calculations when the variance is small
# Default value: 1e-5 (0.00001)
#
EPSILON = 1e-5

#
# EXP_BASE
# The value `n` that is raised to 2i / d_model in the Positional Encoding formula. This value is an arbitarily chosen
# hyperperameter. The authors of "Attention is All You Need" chose 10000 for their n-value.
# Default value: 10000
EXP_BASE = 10000

#
# LEARNING_RATE
# A hyperparameter that determines how much the weights of the neural network are adjusted with respect to the loss
# gradient after each iteration.
# Default value: 0.001
#
LEARNING_RATE = 0.0001

#
# MAX_SEQUENCE_LEN
# The maximum number of tokens that can be input into the transformer per prompt.
# Default value: 5000
#
MAX_SEQUENCE_LEN = 5000