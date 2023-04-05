

def flatten_dict(orig_dict, specific_names):
    """
    Flatten a given dictionary and breakdown nested instances with specific names given.
    :param orig_dict: Dictionary containing nested content
    :param specific_names: List of names for nested elements (must be same size as nested dictionaries in orig_dict)
    :return: Scalar dictionary
    """
    scalar = {}
    for k in orig_dict:
        tensor = orig_dict[k]
        if tensor.ndim == 0:  # Scalar
            scalar[k] = tensor
        else:
            for i, cls in enumerate(specific_names):
                scalar[k + '_' + cls] = tensor[i]

    return scalar
