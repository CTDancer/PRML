"""
loss functions

ref: https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/loss.py
"""
import os
import sys
sys.path.append(os.getcwd())

import numpy as np
import torch
import torch.nn.functional as F
from .base import Loss

class BCELoss(Loss):

    def __init__(self, model, reduction = 'mean'):
        self.input = None
        self.target = None

        self.model = model
        self.reduction = reduction

    def forward(self, input, target):
        self.input = input
        self.target = target

        ###########################################################################
        # TODO:                                                                   #
        # Implement the forward method.                                           #
        ###########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # print('input: ', input)
        # print('target: ', target)
        input = 1 / (1+np.exp(-input))
        epsilon = 1e-7
        loss = -np.mean(target * np.log(input + epsilon) + (1 - target) * np.log(1 - input + epsilon))

        # print('loss: ', loss)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return loss

    def backward(self):
        ###########################################################################
        # TODO:                                                                   #
        # Implement the backward method.                                          #
        # You may use the self.model                                              #
        ###########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        input_grad = (self.input - self.target) / (self.input * (1 - self.input) + 1e-7)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        self.model.backward(input_grad)


class CrossEntropyLoss(Loss):

    def __init__(self, model, reduction = 'mean') -> None:
        self.input = None
        self.target = None

        self.model = model
        self.reduction = reduction

    def forward(self, input, target):
        self.input = input
        self.target = target

        ###########################################################################
        # TODO:                                                                   #
        # Implement the forward method.                                           #
        ###########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        input = torch.from_numpy(input)
        input = F.softmax(input, dim=1)
        target = target.unsqueeze(1).expand_as(input)
        loss = -torch.sum(target * torch.log(input))

        # print('loss: ', loss)
        # print('target: ', target)

        if self.reduction == 'mean':
            loss /= input.size(0)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return loss

    def backward(self):
        ###########################################################################
        # TODO:                                                                   #
        # Implement the backward method.                                          #
        # You may use the self.model                                              #
        ###########################################################################

        input_grad = -self.target.unsqueeze(1) / self.input

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        self.model.backward(input_grad)
