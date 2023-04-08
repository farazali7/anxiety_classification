from src.utils.feature_extraction.features import spectogram
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from src.utils.feature_extraction.features import spectogram


def feature_set_1(data, labels, args):
    """
    Compute STFT images from data.
    :param data: Array of data
    :param labels: Array of respective labels
    :param args: Dictionary of arguments such as sampling_freq, standardize, etc.
    :return: Tuple of features and processed labels
    """
    sampling_freq = args['sampling_freq']
    standardize = args['standardize']

    normalize = False
    if standardize:
        ss = StandardScaler()
        data = [ss.fit_transform(data[..., i]) for i in range(data.shape[-1])]
        data = np.stack(data, axis=-1)
    else:
        normalize = True

    # Compute STFT
    window = 'hamming'
    freq, t, signal = spectogram(data, sampling_freq=sampling_freq, window=window, nperseg=123, n_overlap=0, axis=0)
    signal = np.transpose(signal, [2, 1, 0, 3])  # [trials, channels, freq, time]

    if normalize:
        normalized_signal = np.zeros_like(signal)
        ss = MinMaxScaler()
        for i in range(signal.shape[1]):  # For each channel
            channel_data = signal[:, i, ...]
            scalars = channel_data.reshape(channel_data.shape[0], -1)
            scaled = ss.fit_transform(scalars)
            reshaped_back = scaled.reshape(channel_data.shape)
            normalized_signal[:, i, ...] = reshaped_back
        signal = normalized_signal

    return signal, labels
