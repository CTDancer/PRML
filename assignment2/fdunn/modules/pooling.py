"""
Pooling

ref: https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/pooling.py
"""
import os
import sys
sys.path.append(os.getcwd())

import numpy as np
from .base import Module

class MaxPool2d(Module):
    """Applies a 2D max pooling over an input signal composed of several input
    planes.

    In the simplest case, the output value of the layer with input size :math:`(N, C, H, W)`,
    output :math:`(N, C, H_{out}, W_{out})` and :attr:`kernel_size` :math:`(kH, kW)`
    can be precisely described as:

    .. math::
        \begin{aligned}
            out(N_i, C_j, h, w) ={} & \max_{m=0, \ldots, kH-1} \max_{n=0, \ldots, kW-1} \\
                                    & \text{input}(N_i, C_j, \text{stride[0]} \times h + m,
                                                   \text{stride[1]} \times w + n)
        \end{aligned}

    Args:
        kernel_size: the size of the window to take a max over
        stride: the stride of the window. Default value is :attr:`kernel_size`

    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})` or :math:`(C, H_{in}, W_{in})`
        - Output: :math:`(N, C, H_{out}, W_{out})` or :math:`(C, H_{out}, W_{out})`, where

          .. math::
              H_{out} = \left\lfloor\frac{H_{in} + 2 * \text{padding[0]} - \text{dilation[0]}
                    \times (\text{kernel\_size[0]} - 1) - 1}{\text{stride[0]}} + 1\right\rfloor

          .. math::
              W_{out} = \left\lfloor\frac{W_{in} + 2 * \text{padding[1]} - \text{dilation[1]}
                    \times (\text{kernel\_size[1]} - 1) - 1}{\text{stride[1]}} + 1\right\rfloor
    """
    def __init__(
            self,
            kernel_size,
            stride
        ):
        # input and output
        self.input = None
        self.kernel_size = kernel_size
        self.stride = stride

        self.output_height = None
        self.output_width = None
        self.channels = None

    def forward(self, input):
        self.input = input
        ###########################################################################
        # TODO:                                                                   #
        # Implement the forward method.                                           #
        ###########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # print("pooling forward")

        batch_size, channels, input_height, input_width = input.shape

        # Compute output dimensions
        output_height = (input_height - self.kernel_size) // self.stride + 1
        output_width = (input_width - self.kernel_size) // self.stride + 1

        self.output_height = output_height
        self.output_width = output_width
        self.channels = channels

        # Initialize output tensor
        output = np.zeros((batch_size, channels, output_height, output_width))

        # Perform max pooling
        for b in range(batch_size):
            for c in range(channels):
                for i in range(output_height):
                    for j in range(output_width):
                        # Extract the current pooling window
                        window = input[b, c, i*self.stride:i*self.stride+self.kernel_size,
                                       j*self.stride:j*self.stride+self.kernel_size]
                        # Take the maximum value from the window and store it in the output
                        output[b, c, i, j] = np.max(window)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return output

    def backward(self, output_grad):
        ###########################################################################
        # TODO:                                                                   #
        # Implement the backward method.                                          #
        ###########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # print("pooling backward")

        # print('output_grad.shape: ', output_grad.shape)
        batch_size = output_grad.shape[0]
        channels = self.channels
        output_height = self.output_height
        output_width = self.output_width

        output_grad = output_grad.reshape((batch_size, channels, output_height, output_width))

        input_grad = np.zeros_like(self.input)

        for b in range(batch_size):
            for c in range(channels):
                for i in range(output_height):
                    for j in range(output_width):
                        # Find the indices of the maximum value in the corresponding input window
                        window = self.input[b, c, i*self.stride:i*self.stride+self.kernel_size,
                                            j*self.stride:j*self.stride+self.kernel_size]
                        max_indices = np.unravel_index(np.argmax(window), window.shape)

                        # Set the gradient of the maximum value to be the gradient of the output
                        input_grad[b, c, i*self.stride+max_indices[0], j*self.stride+max_indices[1]] = output_grad[b, c, i, j]

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return input_grad
