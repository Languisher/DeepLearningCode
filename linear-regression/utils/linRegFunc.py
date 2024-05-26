import torch
from torch import Tensor
from typing import *


def linreg(X: Tensor, w: Tensor, b: Tensor | float) -> Tensor:
    """Linear Regression Model
    
    Args:
        X: features
        w: params
        b: bias
        
    Return:
        Result of linreg calculation: X.w + b
    """
    return torch.matmul(X, w) + b

def squared_loss(y_hat: Tensor, 
                 y: Tensor,
                 ) -> Tensor:
    """Calculate squared loss
    
    Args:
        y_hat: predicted result
        y: final result
        
    Return:
        loss value
    """

    return (y_hat - y.reshape_as(y_hat)) ** 2 / 2

def sgd(params: List[Any],
        lr: float,
        batch_size: int) -> None:
    """Stochastic Gradient Descent
        
    Args:
        params: to be performed with gradient descent
        lr: learning rate
        batch_size: size of batch
        
    Return:
        No return
    """
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()