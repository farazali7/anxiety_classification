import random

import matplotlib.pyplot as plt
import numpy as np
import os

from classification.src.config import cfg
from classification.src.utils.data_pipeline import load_data, load_and_concat, convert_to_full_paths


def plot_signal(signal_data, time_vec=None, x_label='', y_label='', title=''):
    """
    Plot signal data.
    :param signal_data: Array of signal data
    :param time_vec: Array of time vector or if None defaults to no. of samples in signal_data
    :param x_label: String for x-axis label
    :param y_label: String for y-axis label
    :param title: String for plot title
    """
    if len(signal_data.shape) > 1:
        signal_data = np.squeeze(signal_data)
    if not time_vec:
        time_vec = np.arange(signal_data.shape[0])
    elif len(time_vec.shape) > 1:
        time_vec = np.squeeze(time_vec)

    plt.plot(time_vec, signal_data)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.title(title)
    plt.show()


def plot_batch_signal(signal_data, time_vec=None, x_label='', y_label='', title_base='', n_subjects=6):
    """
    Plot a batch of subjects' signal data
    :param signal_data: Dict of arrays with signal data per subject as keys
    :param time_vec: List of arrays of time vectors or if None defaults to no. of samples per signal_data
    :param x_label: String for x-axis label
    :param y_label: String for y-axis label
    :param title_base: String for plot title base (will be appended by subject info per plot)
    :param n_subjects: Int for number of subjects to display data from
    """
    # Find rows and cols
    rows = int(np.floor(np.sqrt(n_subjects)))
    cols = n_subjects // rows
    fig, axs = plt.subplots(rows, cols, figsize=[9.6, 7.2])

    samples_on_x = time_vec is None
    for i, key in enumerate(signal_data.keys()):
        row = i // cols
        col = i % cols
        ax = axs[row, col]
        if samples_on_x:
            time_vec = np.arange(signal_data[key].shape[0])
        ax.plot(time_vec, signal_data[key])
        ax.set(xlabel=x_label, ylabel=y_label, title=key)
        # ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    fig.tight_layout(pad=0.8)
    fig.suptitle(title_base)
    plt.subplots_adjust(top=0.87, hspace=0.4)
    plt.show()


def segment_by_grasp(emg_data, labels, grasp_id):
    """
    Return subset of EMG data based on position idxs of desired grasp
    :param emg_data: Array of emg data
    :param labels: Array of graps labels
    :param grasp_id: Int representing ID of desired grasp
    :returns: Array of emg data corresponding to desired grasp
    """
    grasp_idxs = np.where(labels == grasp_id)[0]
    return emg_data[grasp_idxs, :]


def get_batch_grasp_data(dir_path, grasp_id, subject_ids, channels=None, n_subjects=6):
    """
    Get a batch of n_subjects' EMG data pertaining to desired grasp
    :param dir_path: String path to data directory
    :param grasp_id: Int representing ID of desired grasp
    :param subject_ids: List of string IDs for subjects to randomly sample from
    :param channels: Electrode indices to take, if None all are returned
    :param n_subjects: Int for number of subjects to display data from
    :return: Dictionary of relevant EMG grasp data of subjects
    """
    random.seed(7)
    subjects_to_retrieve = random.sample(subject_ids, n_subjects)
    batch_data = {}

    for id in subjects_to_retrieve:
        file_path = os.path.join(dir_path, id + '.pkl')
        emg, labels = load_data(file_path)
        grasp_emg = segment_by_grasp(emg, labels, grasp_id)
        if channels is not None:
            grasp_emg = grasp_emg[:, channels]
        batch_data[id] = grasp_emg

    return batch_data


if __name__ == '__main__':
    CHANNELS = [0]  # Just visualize one channel for now

    # Load samples from NinaPro DB10
    np_cfg = cfg['DATASETS']['NINAPRO_DB10']
    np_processed_data_path = np_cfg['FORMATTED_DATA_PATH']
    np_healthy_subjects = np_cfg['HEALTHY_SUBJECTS']
    np_healthy_subjects = convert_to_full_paths(np_healthy_subjects, np_processed_data_path)
    np_affected_subjects = np_cfg['AFFECTED_SUBJECTS']
    np_affected_subjects = convert_to_full_paths(np_affected_subjects, np_processed_data_path)
    np_x, np_y = load_and_concat(np_healthy_subjects, ext='.pkl')

    gm_cfg = cfg['DATASETS']['GRABMYO']
    gm_processed_data_path = gm_cfg['FORMATTED_DATA_PATH']
    gm_healthy_subjects = gm_cfg['HEALTHY_SUBJECTS']
    gm_healthy_subjects = ['S' + str(x + 115) for x in gm_healthy_subjects]
    gm_healthy_subjects = convert_to_full_paths(gm_healthy_subjects, gm_processed_data_path)
    gm_x, gm_y = load_and_concat(gm_healthy_subjects, ext='.pkl')

    print('Done')
