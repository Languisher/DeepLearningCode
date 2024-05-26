####################################
# 2. Read the dataset
####################################

import torch
import random
from typing import Tuple, Iterator

def data_iter(batch_size: int, 
              features: torch.Tensor, 
              labels: torch.Tensor,
              ) -> Tuple[Iterator[torch.Tensor], Iterator[torch.Tensor]]:
    """Select a batch among all the features
    
    Args:
        batch_size: The size of the batch that we want
        features: The input data
        labels: Labels of input data
    
    Returns:
        features[batch_indices]: A (random) batch of features
        labels[batch_indices]: A (random) batch of corresponding labels of features 
    """
    num_examples = len(features)
    indices = list(range(num_examples))

    random.shuffle(indices)

    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)]
        )
        yield features[batch_indices], labels[batch_indices]