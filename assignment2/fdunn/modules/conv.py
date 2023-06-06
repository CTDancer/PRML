"""
Conv2D

ref: https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/conv.py
"""
import os
import sys
sys.path.append(os.getcwd())

import numpy as np
from .base import Module

class Conv2d(Module):
    """Applies a 2D convolution over an input signal composed of several input
    planes.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int, tuple or str, optional): Padding added to all four sides of
            the input. Default: 0
        padding_mode (str, optional): ``'zeros'``, ``'reflect'``,
            ``'replicate'`` or ``'circular'``. Default: ``'zeros'``
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input
            channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the
            output. Default: ``True``

    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})` or :math:`(C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})` or :math:`(C_{out}, H_{out}, W_{out})`, where

          .. math::
              H_{out} = \left\lfloor\frac{H_{in}  + 2 \times \text{padding}[0] - \text{dilation}[0]
                        \times (\text{kernel\_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor

          .. math::
              W_{out} = \left\lfloor\frac{W_{in}  + 2 \times \text{padding}[1] - \text{dilation}[1]
                        \times (\text{kernel\_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor

    Attributes:
        weight (Tensor): the learnable weights of the module of shape
            :math:`(\text{out\_channels}, \frac{\text{in\_channels}}{\text{groups}},`
            :math:`\text{kernel\_size[0]}, \text{kernel\_size[1]})`.
            The values of these weights are sampled from
            :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
            :math:`k = \frac{groups}{C_\text{in} * \prod_{i=0}^{1}\text{kernel\_size}[i]}`
        bias (Tensor):   the learnable bias of the module of shape
            (out_channels). If :attr:`bias` is ``True``,
            then the values of these weights are
            sampled from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
            :math:`k = \frac{groups}{C_\text{in} * \prod_{i=0}^{1}\text{kernel\_size}[i]}`
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride = 1,
            padding = 0,
            bias = True
    ):
        # input and output
        self.input = None
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # params
        self.params = {}
        ###########################################################################
        # TODO:                                                                   #
        # Implement the params init.                                              #
        ###########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        self.params['W'] = np.random.randn(out_channels, in_channels, kernel_size, kernel_size)
        if bias:
            self.params['b'] = np.random.randn(out_channels)
        else:
            self.params['b'] = None

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # grads of params
        self.grads = {}

    def forward(self, input):
        self.input = input
        ###########################################################################
        # TODO:                                                                   #
        # Implement the forward method.                                           #
        ###########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # print("conv forward")

        batch_size, in_channels, height, width = input.shape
        out_channels, _, kernel_size, _ = self.params['W'].shape

        # Calculate output dimensions
        output_height = (height + 2 * self.padding - kernel_size) // self.stride + 1
        output_width = (width + 2 * self.padding - kernel_size) // self.stride + 1

        # Create output tensor
        output = np.zeros((batch_size, out_channels, output_height, output_width))

        # Apply convolution
        padded_input = np.pad(input, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        for b in range(batch_size):
            for c_out in range(out_channels):
                for c_in in range(in_channels):
                    for i in range(output_height):
                        for j in range(output_width):
                            output[b, c_out, i, j] += np.sum(padded_input[b, c_in, i*self.stride:i*self.stride+kernel_size, j*self.stride:j*self.stride+kernel_size] * self.params['W'][c_out, c_in])

        # Add bias if enabled
        if self.params['b'] is not None:
            output += self.params['b'].reshape((1, out_channels, 1, 1))

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return output

    def backward(self, output_grad):
        ###########################################################################
        # TODO:                                                                   #
        # Implement the backward method.                                          #
        ###########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # print("conv backward")

        batch_size, out_channels, output_height, output_width = output_grad.shape
        _, in_channels, kernel_size, _ = self.params['W'].shape

        # Initialize gradients
        input_grad = np.zeros_like(self.input)
        self.grads['W'] = np.zeros_like(self.params['W'])
        if self.params['b'] is not None:
            self.grads['b'] = np.zeros_like(self.params['b'])

        # Calculate gradients
        padded_input = np.pad(self.input, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        padded_input_grad = np.pad(input_grad, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        for b in range(batch_size):
            for c_out in range(out_channels):
                for c_in in range(in_channels):
                    for i in range(output_height):
                        for j in range(output_width):
                            input_grad[b, c_in, i*self.stride:i*self.stride+kernel_size, j*self.stride:j*self.stride+kernel_size] += output_grad[b, c_out, i, j] * self.params['W'][c_out, c_in]
                            self.grads['W'][c_out, c_in] += output_grad[b, c_out, i, j] * padded_input[b, c_in, i*self.stride:i*self.stride+kernel_size, j*self.stride:j*self.stride+kernel_size]

        # Compute bias gradients if enabled
        if self.params['b'] is not None:
            self.grads['b'] = np.sum(output_grad, axis=(0, 2, 3))

        # Remove padding from input gradients
        input_grad = input_grad[:, :, self.padding:-self.padding, self.padding:-self.padding]

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return input_grad
