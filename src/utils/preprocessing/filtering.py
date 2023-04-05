from scipy.signal import iirnotch, butter, sosfiltfilt, filtfilt


def butter_bandpass_filter(data, order=4, cutoffs=None, sampling_freq=100, axis=0):
    """
    Perform bandpass filtering using a butterworth filter.
    :param order: Int for butterworth filter order
    :param cutoffs: Tuple of low-pass and high-pass cutoff frequencies
    :param sampling_freq: Integer for sampling frequency of data
    :param axis: Int for axis to filter on
    :return: Butterworth bandpass filtered data
    """
    if cutoffs is None:
        cutoffs = [20, 125]  # Default low and high

    sos = butter(order, cutoffs, 'bandpass', fs=sampling_freq, output='sos')
    filtered = sosfiltfilt(sos, data, axis=axis)

    return filtered


def butter_highpass_filter(data, order=4, cutoff=20, sampling_freq=100, axis=0):
    """
    Perform bandpass filtering using a butterworth filter.
    :param order: Int for butterworth filter order
    :param cutoff: Int for High-pass cutoff
    :param sampling_freq: Integer for sampling frequency of data
    :param axis: Int for axis to filter on
    :return: Butterworth bandpass filtered data
    """
    sos = butter(order, cutoff, 'highpass', fs=sampling_freq, output='sos')
    filtered = sosfiltfilt(sos, data, axis=axis)

    return filtered


def notch_filter(data, freq=60, qf=30.0, sampling_freq=100, axis=0):
    """
    Perform notch filtering.
    :param data: Array of data
    :param freq: Frequency to remove from data
    :param qf: Quality factor, represents the ratio between center frequency and bandwidth (high for notch)
    :param sampling_freq: Integer for sampling frequency of data
    :param axis: Int for axis to filter on
    :return: Data with freq removed
    """
    b, a = iirnotch(freq, qf, sampling_freq)
    filtered = filtfilt(b, a, data, axis=axis)

    return filtered
