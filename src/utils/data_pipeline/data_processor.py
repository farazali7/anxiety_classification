import os
import numpy as np
from src.utils.preprocessing import butter_bandpass_filter, butter_highpass_filter, notch_filter, \
    downsample
from src.utils.data_pipeline import save_data, load_data
from src.config import cfg
from src.constants import FEATURE_EXTRACTION_FUNCS

from tqdm import tqdm
from functools import partial
from multiprocessing import Pool


def preprocess_dreamer_data(data, labels, butter_ord, butter_freq, sampling_freq, save_path=None):
    """
    Preprocess the data by filtering and restructuring class labels etc.
    :param data: Array of data
    :param labels: Labels vector, might be None
    :param butter_ord: Integer for butterworth filter order
    :param butter_freq: Cutoff frequency(ies)
    :param sampling_freq: Frequency data was sampled at
    :param save_path: String, path to save the data in
    """
    # For each trial
    all_trial_eeg = []
    all_trial_labels = []
    for trial in data.keys():
        trial_data = data[trial]
        trial_eeg, trial_baseline, trial_arousal, trial_valence = trial_data.values()

        # Take last 64 seconds
        trial_eeg = trial_eeg[-sampling_freq*64:, :]

        # Common average reference
        car = np.mean(trial_eeg, axis=-1)[..., np.newaxis]
        car = np.tile(car, trial_eeg.shape[-1])
        trial_eeg -= car

        # Band-pass filter
        trial_eeg = butter_bandpass_filter(trial_eeg, butter_ord, butter_freq, sampling_freq)

        # Trim 2 seconds from each side; 60 second stimulus now
        trial_eeg = trial_eeg[2*sampling_freq:-2*sampling_freq, :]

        # Split into 1 second segments
        seg_len = 1*sampling_freq
        trial_eeg_segments = [trial_eeg[i:i+seg_len] for i in range(0, len(trial_eeg), seg_len)]

        # Baseline removal
        # Takeout first 2 and last 2 seconds of baseline
        trial_baseline = trial_baseline[2*sampling_freq:-2*sampling_freq, :]
        baseline_segments = [trial_baseline[i:i+seg_len] for i in range(0, len(trial_baseline), seg_len)]
        baseline_mean = np.mean(baseline_segments)

        trial_eeg_segments = [eeg_seg - baseline_mean for eeg_seg in trial_eeg_segments]
        trial_eeg = np.vstack(trial_eeg_segments)

        all_trial_eeg.append(trial_eeg)

        # Get LVLA, LVHA, HVLA, and HVHA labels
        # Just one score assigned for each trial
        arousal_score = np.unique(trial_arousal)
        valence_score = np.unique(trial_valence)

        threshold = 3
        if arousal_score < threshold and valence_score < threshold:  # LVLA
            quadrant = 0
        elif arousal_score > threshold and valence_score < threshold:  # LVHA
            quadrant = 1
        elif arousal_score < threshold and valence_score > threshold:  # HVLA
            quadrant = 2
        else:  # HVHA
            quadrant = 3

        all_trial_labels.append(quadrant)

    eeg_data = np.stack(all_trial_eeg, axis=-1)
    labels = np.stack(all_trial_labels, axis=-1)

    if save_path:
        save_data(eeg_data, labels, save_path)

    return eeg_data, labels


def extract_features(data, labels, feature_extraction_func, feature_extraction_args, save_path=None):
    '''
    Retrieve features extracted for a given data array.

    :param save_path: String, path to save the data in
    :return: Tuple of two NumPy arrays as (features, labels)
    '''

    features, labels = feature_extraction_func(data, labels, feature_extraction_args)

    if save_path:
        save_data(features, labels, save_path)

    return features, labels


def process_data(subject_id, data_dir, preprocess_func, preprocessing_args, feature_extraction_func,
                 feature_extraction_args, save_dir=None):
    """
    Process a subject's data by preprocessing (filtering) and extracting features (+windowing).
    :param subject_id: String, subject ID specified by dataset file naming for specific subject
    :param data_dir: String, path to main dataset directory
    :param preprocess_func: Function for preprocessing
    :param preprocessing_args: Dictionary of keyword args for preprocessing
    :param feature_extraction_func: Function for feature extraction
    :param feature_extraction_args: Dictionary of keyword args for feature extraction
    :param save_dir: String, path to directory to save the data in
    :return: Tuple of processed data and labels
    """
    data_path = os.path.join(data_dir, str(subject_id) + '.pkl')
    data, labels = load_data(data_path)

    # Preprocessing
    preprocessed_data, preprocessed_labels = preprocess_func(data, labels, **preprocessing_args)

    # Feature extraction
    features, feature_labels = extract_features(preprocessed_data, preprocessed_labels,
                                                feature_extraction_func, feature_extraction_args)

    if save_dir:
        save_path = os.path.join(save_dir, subject_id + '.pkl')
        save_data(features, feature_labels, save_path)

    return features, feature_labels


if __name__ == '__main__':
    # Preprocessing args
    butter_ord = cfg['BUTTERWORTH_ORDER']
    butter_freq = cfg['BUTTERWORTH_FREQ']

    preprocessing_args = {'butter_ord': butter_ord,
                          'butter_freq': butter_freq}

    # Feature extraction args
    standardize = cfg['STANDARDIZE']
    feature_extraction_func = FEATURE_EXTRACTION_FUNCS[cfg['FEATURE_EXTRACTION_FUNC']]

    feature_extraction_args = {'standardize': standardize}

    # DREAMER
    dm_cfg = cfg['DATASETS']['DREAMER']
    formatted_data_path = dm_cfg['FORMATTED_DATA_PATH']
    subject_ids = dm_cfg['SUBJECTS']
    dm_sampling_freq = dm_cfg['SAMPLING_FREQ']
    save_dir = dm_cfg['PROCESSED_DATA_PATH']

    preprocessing_args['sampling_freq'] = dm_sampling_freq
    feature_extraction_args['sampling_freq'] = dm_sampling_freq

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    process_data(subject_ids[0], data_dir=formatted_data_path,
                                          preprocess_func=preprocess_dreamer_data,
                                          preprocessing_args=preprocessing_args,
                                          feature_extraction_func=feature_extraction_func,
                                          feature_extraction_args=feature_extraction_args,
                                          save_dir=save_dir)

    with Pool() as pool:
        res = list(tqdm(pool.imap(partial(process_data,
                                          data_dir=formatted_data_path,
                                          preprocess_func=preprocess_dreamer_data,
                                          preprocessing_args=preprocessing_args,
                                          feature_extraction_func=feature_extraction_func,
                                          feature_extraction_args=feature_extraction_args,
                                          save_dir=save_dir), subject_ids),
                        total=len(subject_ids)))

    print('Done.')
