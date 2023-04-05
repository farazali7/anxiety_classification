import numpy as np
import scipy


def homogenize_window(window_data):
    '''
    Homogenize data_pipeline in a window based on the mode (most frequently occurring value). Useful for ground-truth windows.
    :param window_data: Windowed data_pipeline with >= 2 dims
    :return: New array containing mode across each row
    '''

    return scipy.stats.mode(np.squeeze(window_data), 1, keepdims=True).mode