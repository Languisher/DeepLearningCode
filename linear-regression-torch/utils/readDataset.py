from torch.utils import data


def load_array(data_arrays,
               batch_size,
               is_train=True):
    """
    Divide input_data into small batches, and yield them one by one.
    
    Args:
        data_arrays: containing all (features, labels)
        batch_size: size of batch
    """
    
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset,
                           batch_size,
                           shuffle=is_train)