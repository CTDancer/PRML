"""
Linear Layer

ref: https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/linear.py
"""
import os
import sys
sys.path.append(os.getcwd())

import numpy as np
from .base import Module

class Linear(Module):
    def __init__(self, in_features, out_features, bias = True):

        # input and output
        self.input = None
        self.in_features = in_features
        self.out_features = out_features

        # params
        self.params = {}
        k= 1/in_features
        self.params['W'] = np.random.uniform(low=-np.sqrt(k), high=np.sqrt(k), size=(out_features,in_features))
        self.params['b'] = None
        if bias:
            self.params['b'] = np.random.uniform(low=-np.sqrt(k), high=np.sqrt(k), size=(out_features))

        # grads of params
        self.grads = {}

    def forward(self, input):
        self.input = input
        ###########################################################################
        # TODO:                                                                   #
        # Implement this method. math:  Y = XW^T + b                              #
        ###########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        print(input.shape)
        print(self.in_features)

        input_reshaped = input.reshape(-1, self.in_features)  # 调整输入矩阵的形状
        print(input_reshaped.shape)
        print(self.params['W'].shape)
        output = np.matmul(input_reshaped, self.params['W'].T)
        print("here")
        if self.params['b'] is not None:
            output += self.params['b']

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return output

    def backward(self, output_grad):
        ###########################################################################
        # TODO:                                                                   #
        # Calculate and store the grads of self.params['W'] and self.params['b']  #
        # in self.grads['W'] and self.grads['b'] respectively.                    #
        # Calculate and return the input_grad.                                    #
        # Notice: You have to deal with high-dimensional tensor inputs            #
        ###########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        self.grads['W'] = np.matmul(output_grad.T, self.input).T
        self.grads['b'] = np.sum(output_grad, axis=0)
        input_grad = np.matmul(output_grad, self.params['W'])

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        assert self.grads['W'].shape == self.params['W'].shape
        assert self.grads['b'].shape == self.params['b'].shape
        assert input_grad.shape == self.input.shape

        return input_grad

def unit_test():
    np.random.seed(2333)

    model = Linear(20,30)
    input = np.random.randn(4, 2, 8, 20)
    output = model(input)
    print (output.shape)

    output_grad = output
    input_grad = model.backward(output_grad)
    print (model.grads['W'].shape)
    print (model.grads['b'].shape)
    print (input_grad.shape)

if __name__ == '__main__':
    unit_test()