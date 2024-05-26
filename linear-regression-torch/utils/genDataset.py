####################################
# 1. Generation of the dataset
####################################

import torch
from typing import Tuple

def synthetic_data(w: torch.Tensor, b: float, num_examples: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate synthetic data.
    
    Args:
        w: weight
        b: bias
        num_examples: number of examples

    Returns:
        X: Generated data (Features)
        y: Generated labels
    """

    X = torch.normal(0, 1, (num_examples, len(w))) # shape of X should be (d, n), to multiply with size of w (n, 1)
    y = torch.matmul(X, w).reshape((-1, 1))
    noise = torch.normal(0, 0.01, y.shape)
    y += noise

    return X, y
    

if __name__ == '__main__':
    w = torch.Tensor([2, 3])
    b = 4.2
    num_examples = 1000
    features, labels = synthetic_data(w, b, 100)
    synthetic_data(w, b, num_examples)