"""Implements a Convolutional Neural Network(CNN) for the detection of facial landmarks"""


import torch
import torch.nn as nn
import torch.nn.functional as F


__author__ = "Victor Mawusi Ayi <ayivima@hotmail.com>"


class Net(nn.Module):

    def __init__(self):
        """Initializes the neural network state"""
        
        super(Net, self).__init__()
        
        # starting out with very few output channels
        # is good for efficiency and might be a first step
        # to preventing overfitting
        self.conv1 = nn.Conv2d(1, 8, 5, 1, 2)
        self.conv2 = nn.Conv2d(8, 16, 5, 1, 2)
        self.conv3 = nn.Conv2d(16, 32, 5, 2)
        self.conv4 = nn.Conv2d(32, 64, 3, 1, 1)
        self.conv5 = nn.Conv2d(64, 64, 3, 1, 1)
        
        self.dense1 = nn.Linear(6*6*64, 256)
        self.dense2 = nn.Linear(256, 136)

        self.pool = nn.MaxPool2d(2, 2)
        self.drop = nn.Dropout(p=0.3)        

        
    def forward(self, x):
        """Implements the forward pass of an image tensor through the neurons.
        
        Arguments
        ---------
        :x: an image tensor
        """
        
        x = F.selu(self.conv1(x))
        x = self.pool(x)
        x = F.selu(self.conv2(x))
        x = self.pool(x)
        x = F.selu(self.conv3(x))
        x = F.selu(self.conv4(x))
        x = self.pool(x)
        x = F.selu(self.conv5(x))
        x = self.pool(x)

        x = x.view(x.size(0), -1)

        x = self.drop(F.selu(self.dense1(x)))
        
        # selu is self normalizing and it did serve a good purpose,
        # even though it looks odd that the outputs of the last layer 
        # gets passed through an activation. 
        x = F.selu(self.dense2(x))

        return x
