import logging
from random import Random
from itertools import zip_longest, cycle, islice
import math
import os
import torch


def create_equal_folds(data_sources, num_folds=5, seed=None, save_dir=None):
    """
    Partition data into folds containing equal amounts of data by specified groups.
    :param data_sources: List of data sources which will be equally split amongst folds
    :param num_folds: Int specifying number of desired folds
    :param seed: Int for setting random seed
    :param save_dir: Path to save fold splits to
    :return: Folds that contain equally distributed data from all data_sources
    """
    rand = Random(seed)
    data_sizes = [len(data_source) for data_source in data_sources]
    qts_in_fold = [ds_size // num_folds if ds_size >= num_folds else 1 for ds_size in data_sizes]

    subsets = []
    for i, data_source in enumerate(data_sources):
        size = data_sizes[i]
        quantity = qts_in_fold[i]
        subsets.append([data_source[j:j+quantity] for j in range(0, size, quantity)])

    all_folds = [*zip_longest(*subsets)]
    desired_folds = all_folds[:num_folds]
    remainder_folds = all_folds[num_folds:]

    # Flatten remainder
    flattened_remainders = []
    for rem_fold in remainder_folds:
        for i, rem_source in enumerate(rem_fold):
            if rem_source is not None:
                flattened_remainders += rem_source

    # Flatten desired
    flattened_desired = []
    for desired_fold in desired_folds:
        flattened_fold = []
        for source in desired_fold:
            if source is not None:
                flattened_fold += source
        flattened_desired.append(flattened_fold)

    # Work backwards (to start with most sparse folds) to evenly distribute remainder folds
    for i in range(len(flattened_remainders)):
        desired_idx = len(flattened_desired) - (1 + (i % num_folds))
        flattened_desired[desired_idx].append(flattened_remainders[i])

    # Remove None filler values
    final_folds = []
    for fold in flattened_desired:
        filtered_fold = [val for val in fold if val is not None]
        rand.shuffle(filtered_fold)
        final_folds.append(filtered_fold)

    rand.shuffle(final_folds)

    if save_dir:
        torch.save(final_folds, os.path.join(save_dir, 'folds.pkl'))

    return final_folds


def create_equal_folds_by_percentage(data_sources, percentage=0.1, seed=None):
    """
    Create folds containing equal amounts of data from all data_sources where each fold has roughly a specified
    percentage of the total data.
    :param data_sources: List of data sources which will be equally split amongst folds
    :param percentage: Double specifying percentage of total data each fold should have
    :param seed: Int for setting random seed
    :return: Folds that contain equally distributed data from all data_sources and have specified percent of all data
    """
    num_folds = math.ceil(1/percentage)
    folds = create_equal_folds(data_sources, num_folds, seed)

    return folds


def partition_dataset(data_sources, test_percentage=0.1, seed=None, save_dir=None):
    """
    Partition dataset into train and test sets based on percentage. Train set will be data_sources with test samples
    excluded.
    :param data_sources: List of data sources which will be equally split amongst folds
    :param test_percentage: Double specifying ratio of total data that makes up test set (train is 1-test_percentage)
    :param seed: Int for setting random seed
    :param save_dir: Path for saving test split, train split not saved here
    :return: Tuple of (train data sources with test members removed, test members)
    """
    rand = Random(seed)
    data_sizes = [len(data_source) for data_source in data_sources]
    max_data_size = max(data_sizes)
    total_data_size = sum(data_sizes)

    # Compute number of required test members
    num_in_test = math.ceil(total_data_size*test_percentage)

    # Shuffle all sources
    for source in data_sources:
        rand.shuffle(source)

    # Round-robin fill until target number of members reached
    test_set = round_robin_fill(data_sources=data_sources,
                                target=num_in_test,
                                max_source_len=max_data_size,
                                remove_from_source=True)

    actual_test_percentage = len(test_set) / total_data_size
    if actual_test_percentage != test_percentage:
        logging.warning(f'Could not create test set with exactly: {round(test_percentage*100,1)}% of data... '
                        f'Created test set with: {round(actual_test_percentage*100,1)}% of data instead.\n')

    if save_dir is not None:
        torch.save(test_set, os.path.join(save_dir, 'test_set.pkl'))

    return data_sources, test_set


def stratified_partition(data_sources, test_percentage=0.1, seed=None):
    """
    Perform stratified partitioning of data sources into train and test sets based on specified percentage.
    :param data_sources: List of data sources which will be equally split amongst folds
    :param test_percentage: Double specifying ratio of total data that makes up test set (train is 1-test_percentage)
    :param seed: Int for setting random seed
    :return: Tuple of stratified (train set, test set) members
    """
    rand = Random(seed)
    data_sizes = [len(data_source) for data_source in data_sources]
    test_qts = [math.ceil(ds_size*test_percentage) for ds_size in data_sizes]

    train_set = []
    test_set = []
    for i, data_source in enumerate(data_sources):
        rand.shuffle(data_source)
        test_set += data_source[:test_qts[i]]
        train_set += data_source[test_qts[i]:]

    actual_test_percentage = len(test_set) / sum(data_sizes)
    if actual_test_percentage != test_percentage:
        logging.warning(f'Could not create test set with exactly: {round(test_percentage*100,1)}% of data... '
                        f'Created test set with: {round(actual_test_percentage*100,1)}% of data instead.\n')

    rand.shuffle(test_set)
    rand.shuffle(train_set)

    return train_set, test_set


def round_robin_fill(data_sources, target, max_source_len, remove_from_source=False):
    """
    Fill a list with values from data_sources in a round-robin manner until list size is target.
    :param data_sources: List of data sources which will be equally split amongst folds
    :param target: Int for desired final list size
    :param max_source_len: Int for maximum length of a data source
    :param remove_from_source: Boolean indicating whether to pop elements from respective sources
    :return: List populated with elements from data_sources
    """
    assert target > 0, 'Variable \'target\' cannot be <= 0.'

    res = []

    while max_source_len > 0 and len(res) < target:
        for i in data_sources:
            if len(i) != 0:
                if remove_from_source:
                    res.append(i.pop(0))
                else:
                    res.append(i[0])
                if len(res) == target:
                    break
        max_source_len -= 1

    if len(res) != target:
        logging.warning(f'Could not fill list with {target} elements, returning list with {len(res)} elements.')

    return res


if __name__ == "__main__":
    test_a = ['A', 'A', 'A', 'A', 'A', 'A']
    test_b = ['B', 'B', 'B', 'B', 'B']
    test_c = ['C', 'C']
    ds = [test_a, test_b, test_c]
    train, test = stratified_partition(ds, 0.1, 4)

    print('done')
