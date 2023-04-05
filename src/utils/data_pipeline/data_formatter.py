import os
from functools import partial
from multiprocessing import Pool

import numpy as np
from scipy.io import loadmat
from tqdm import tqdm

from src.config import cfg
from src.utils.data_pipeline import save_data, save_dict


def format_dreamer_data(subject_id, data_path, save_dir=None):
    '''
    Load and retrieve NumPy arrays containing pertinent EMG classification data_pipeline + labels for given subject from the raw
    dataset.
    :param data_path: String, path to main dataset directory
    :param subject_id: String, subject number specified by dataset file naming for specific subject
    :param data_col: String, column name for data_pipeline
    :param label_col: String, column name for labels,
    :param grasp_ids: List of grasp int IDs
    :param electrode_ids: Array of ints representing IDs for electrode channels
    :param save_dir: String, path to directory to save the data in
    :return Tuple of two NumPy arrays as (emg data_pipeline, grasp labels)
    '''
    data = loadmat(os.path.join(data_path, str(subject_id)+'.mat'))

    arousals = np.squeeze(data['arousal'])
    valences = np.squeeze(data['valence'])
    baselines = data['baseline']

    full_dict = {}
    # Create trial-wise entries
    for i in range(18):
        eeg_data = data['trial'+str(i+1)]
        trial_arousal = np.full_like(eeg_data, fill_value=arousals[i], dtype=np.int32)
        trial_valence = np.full_like(eeg_data, fill_value=valences[i], dtype=np.int32)
        trial_baseline = baselines[..., i]
        full_dict[i] = {
            'eeg': eeg_data,
            'baseline': trial_baseline,
            'arousal': trial_arousal,
            'valence': trial_valence
        }

    if save_dir:
        save_path = os.path.join(save_dir, subject_id + '.pkl')
        save_dict(full_dict, save_path)

    return full_dict


if __name__ == '__main__':
    # DREAMER
    dm_cfg = cfg['DATASETS']['DREAMER']
    raw_data_path = dm_cfg['RAW_DATA_PATH']
    subject_ids = dm_cfg['SUBJECTS']
    save_dir = dm_cfg['FORMATTED_DATA_PATH']

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    format_params = {'data_path': raw_data_path,
                     'save_dir': save_dir}

    with Pool() as pool:
        res = list(tqdm(pool.imap(partial(format_dreamer_data, **format_params), subject_ids),
                        total=len(subject_ids)))

    print('Done.')

