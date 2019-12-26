"""
Implements several combinations of Loss and Optimizers.
"""

from torch.nn import SmoothL1Loss, MSELoss
from torch.optim import Adam


__author__ = "Victor Mawusi Ayi <ayivima@hotmail.com>"


def SmoothL1Adam(learn_params, lr=0.001, amsgrad=True):
    """Returns a SmoothL1 loss function and an Adam optimizer.
    
    Arguments
    ---------
    :learn_params: The parameters to be learned during training
    :lr: The learning rate for the Adam optimizer
    :amsgrad: Used to specify whether to use the AMSGRAD variant 
              instead of the traditional ADAM.
    """
    
    criterion = SmoothL1Loss()
    optimizer = Adam(learn_params, lr=lr, amsgrad=True)
    
    return criterion, optimizer


def MSEAdam(learn_params, lr=0.001, amsgrad=True):
    """Returns a Mean Square Error(MSE) loss function and an Adam optimizer.
    
    Arguments
    ---------
    :learn_params: The parameters to be learned during training
    :lr: The learning rate for the Adam optimizer
    :amsgrad: Used to specify whether to use the AMSGRAD variant 
              instead of the traditional ADAM.
    """
    
    criterion = MSELoss()
    optimizer = Adam(learn_params, lr=lr, amsgrad=True)
    
    return criterion, optimizer
