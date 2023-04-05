import torch


def compute_class_weights(labels):
    """
    Compute weights for each class to use in subsequent loss function.
    :param labels: Torch tensor of data labels with C unique values
    :return: Torch tensor of C weights
    """
    labels = torch.squeeze(labels)
    total = len(labels)
    class_counts = torch.bincount(labels)
    class_weights = 1 - (class_counts/total)

    return class_weights
