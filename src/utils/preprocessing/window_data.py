import numpy as np
from numpy.lib.stride_tricks import as_strided


def window_data(data, window_size=200, overlap_size=100, remove_short=True, flatten_inside_window=False):
    '''
    Windowing function to split data_pipeline based on window_size
    :param data: NumPy array of data_pipeline
    :param window_size: Integer, number of samples in one window
    :param overlap_size: Integer, number of overlapping samples between windows
    :param remove_short: Boolean, set True to remove (last) shorter window
    :param flatten_inside_window: Boolean, set True to flatten window size dimension with outer dimension
    :return: Windowed array view
    '''
    assert data.ndim == 1 or data.ndim == 2
    if data.ndim == 1:
        data = data.reshape((-1, 1))

    # get the number of overlapping windows that fit into the data_pipeline
    num_windows = (data.shape[0] - window_size) // (window_size - overlap_size) + 1
    overhang = data.shape[0] - (num_windows*window_size - (num_windows-1)*overlap_size)

    # if there's overhang, remove last shorter window
    if overhang != 0 and remove_short:
        data = data[:-overhang]

    sz = data.dtype.itemsize
    if not data.data.contiguous:
        data = np.ascontiguousarray(data)
    ret = as_strided(
            data,
            shape=(num_windows, window_size*data.shape[1]),
            strides=((window_size-overlap_size)*data.shape[1]*sz, sz),
            writeable=False
            )

    if flatten_inside_window:
        return ret
    else:
        return ret.reshape((num_windows, -1, data.shape[1]))
