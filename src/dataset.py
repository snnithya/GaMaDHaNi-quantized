from typing import Any, Callable, Dict, Optional, Tuple
from functools import partial

import gin
import inspect
import lmdb
import torch
import pdb
import numpy as np
from torch.utils.data import Dataset, Subset
from protobuf.data_example import AudioExample


TensorDict = Dict[str, torch.Tensor]

@gin.configurable
class Task:
    """
    read_fn: function to read and transform the data sequence in the format the training script accepts
    invert_fn: applying the inverse of the transform applied in read_fn
    common_args: the common arguments required by both the read_fn and the invert_fn
    kwargs in read and invert functions are to pass the dynamic entities i.e. Data
    """
    def __init__(self, read_fn: Callable[..., Any], invert_fn: Callable[..., Any], **kwargs):
        self.read_fn = read_fn
        self.invert_fn = invert_fn
        self.extra_args = kwargs["kwargs"] #because of gin file?

    def read_(self, **kwargs):
        kwargs.update(self.extra_args)
        return self.read_fn(**kwargs)
        
    def invert_(self, **kwargs):
        kwargs.update(self.extra_args)
        return self.invert_fn(**kwargs)

    

#utils
# def plot(f0s, primed=True):
#     if primed:
#         invert(f0_gt)
# def save()

# #generate
# def generate(): #or validation
#     save(invert_fn, plot_fn, save_audio)


def ordered_dataset_split(dataset: Dataset,
                          sizes: Tuple[int, ...]) -> Tuple[Dataset]:
    assert sum(sizes) == len(dataset)
    datasets = []
    start_idx = 0
    for size in sizes:
        datasets.append(Subset(dataset, range(start_idx, start_idx + size)))
        start_idx += size
    return tuple(datasets)


@gin.configurable
class SequenceDataset(Dataset):
    def __init__(
            self,
            db_path: str,
            task: Optional[Task] = None,
            apply_transform: bool = False) -> None:
        super().__init__()
        self._env = None
        self._keys = None
        self._db_path = db_path
        self._task = task
        self._apply_transform = apply_transform

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        with self.env.begin() as txn:
            ae = AudioExample(txn.get(self.keys[index]))
        ae = ae.as_dict()
        if self._task is not None:
            if self._apply_transform:
                if "train" in self._db_path:
                    transpose_pitch = np.random.randint(-400,400)
                    ae = self._task.read_(**{"inputs": ae, "transpose_pitch": transpose_pitch})
                else:
                    ae = self._task.read_(**{"inputs": ae})
            else:
                pdb.set_trace()
                ae = self._task.read_(**{"inputs": ae})
           
        if ae:
            return ae

    @property
    def env(self):
        if self._env is None:
            self._env = lmdb.open(
                self._db_path,
                lock=False,
                readahead=False,
            )
        return self._env

    @property
    def keys(self):
        if self._keys is None:
            with self.env.begin(write=False) as txn:
                self._keys = list(txn.cursor().iternext(values=False))
        return self._keys