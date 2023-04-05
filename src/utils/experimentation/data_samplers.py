import torch
import random
from typing import Iterator, List
from torch.utils.data import BatchSampler, Sampler
import numpy as np


# # combined dataset class
class CombinationDataset(torch.utils.data.DataLoader):
    def __init__(self, datasets):
        self.datasets = datasets

    def __len__(self):
        return (sum([dataset.__len__() for dataset in self.datasets]))

    def __getitem__(self, indices):
        dataset_idx = indices[0]
        data_idx = indices[1]
        print(indices)
        return self.datasets[dataset_idx].__getitem__(data_idx)


# class that will take in multiple samplers and output batches from a single dataset at a time
class ComboBatchSampler:
    def __init__(self, samplers, batch_size, drop_last):
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.samplers = samplers
        self.iterators = [iter(sampler) for sampler in self.samplers]
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[List[int]]:
        # define how many batches we will grab
        self.min_batches = min([len(sampler) for sampler in self.samplers])
        self.n_batches = self.min_batches * len(self.samplers)

        # define which indices to use for each batch
        self.dataset_idxs = []
        random.seed(42)
        for j in range((self.n_batches // len(self.samplers) + 1)):
            loader_inds = list(range(len(self.samplers)))
            random.shuffle(loader_inds)
            self.dataset_idxs.extend(loader_inds)
        self.dataset_idxs = self.dataset_idxs[:self.n_batches]

        # return the batch indices
        batch = []
        for dataset_idx in self.dataset_idxs:
            for idx in self.samplers[dataset_idx]:
                batch.append(idx)
                if len(batch) == self.batch_size:
                    yield (batch)
                    batch = []
                    break
            if len(batch) > 0 and not self.drop_last:
                yield batch

    def __len__(self) -> int:
        if self.drop_last:
            return (sum([len(sampler) for sampler in self.samplers])) // self.batch_size
        else:
            return (sum([len(sampler) for sampler in self.samplers]) + self.batch_size - 1) // self.batch_size


class CustomBatchSampler(Sampler[List[int]]):
    def __init__(self, samplers, batch_size: int, drop_last: bool) -> None:
        # Since collections.abc.Iterable does not check for `__getitem__`, which
        # is one way for an object to be an iterable, we don't do an `isinstance`
        # check here.
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.samplers = samplers
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.sampler_idxs = [i for i in range(len(self.samplers))]
        self.total_size = sum([len(sampler) for sampler in self.samplers])  # Total number of samples
        self.n_batches_by_sampler = [len(sampler)//self.batch_size for sampler in self.samplers]

    def __iter__(self):
        batch = []
        sampler_id = int(np.random.choice(self.sampler_idxs, 1))
        for idx in self.samplers[sampler_id]:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        if self.drop_last:
            return (sum([len(sampler) for sampler in self.samplers])) // self.batch_size
        else:
            return (sum([len(sampler) for sampler in self.samplers]) + self.batch_size - 1) // self.batch_size

    # def __len__(self) -> int:
    #     # Can only be called if self.sampler has __len__ implemented
    #     # We cannot enforce this condition, so we turn off typechecking for the
    #     # implementation below.
    #     # Somewhat related: see NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
    #     if self.drop_last:
    #         return len(self.sampler) // self.batch_size  # type: ignore[arg-type]
    #     else:
    #         return (len(self.sampler) + self.batch_size - 1) // self.batch_size  # type: ignore[arg-type]
