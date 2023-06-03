"""
Activation functions

ref: https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/activation.py
"""

import os
import sys
sys.path.append(os.getcwd())

import numpy as np
from .base import Module

class Sigmoid(Module):
    def __init__(self):
        self.input = None
        self.output = None
        self.params = None

    def forward(self, input):
        ###########################################################################
        # TODO:                                                                   #
        # Implement the forward method.                                           #
        ###########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        output = 1 / (1 + np.exp(-input))

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        self.output = output
        return output

    def backward(self, output_grad):
        ###########################################################################
        # TODO:                                                                   #
        # Implement the backward method.                                          #
        ###########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        input_grad = output_grad * self.output * (1 - self.output)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return input_grad