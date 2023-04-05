import numpy as np


def downsample(data, up, down):
    """
    Resample a given data array to a sampling rate of up/down. NOTE: ASSUMES FILTERED DATA TO AVOID ALIASING.
    :param data: Array of data
    :param up: Int upsampling factor
    :param down: Int downsampling factor
    :param axis: Int axis to resample along
    :return: Data with up/down sampling rate
    """
    curr_total = len(data)
    ratio = down / up
    num_samples_new = int(curr_total / ratio)
    indices = np.round(np.linspace(0, curr_total-1, num=num_samples_new)).astype(int)
    resampled = data[indices, :]

    return resampled
