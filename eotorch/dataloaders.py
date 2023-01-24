from .samplers import RandomBatchSampler
from torch.utils.data import DataLoader as TorchDataLoader


def init_torch_dataloader(dataset, sampler=None, **kwargs):
    key = "batch_sampler"
    if "batch_size" in kwargs:
        if not isinstance(sampler, RandomBatchSampler):
            key = "sampler"
    if sampler is not None:
        sampler.dataset = dataset
    kwargs[key] = sampler
    return TorchDataLoader(dataset, **kwargs)


class DataLoader:

    def __init__(self, dataset=None, sampler=None, **kwargs):
        self.dataset = dataset
        self.sampler = sampler
        self._kwargs = kwargs

    def set_dataset(self, value):
        self.dataset = value

    def set_sampler(self, value):
        self.sampler = value

    def to_torch(self):
        kwargs = self._kwargs.copy()
        key = "batch_sampler"
        if "batch_size" in kwargs:
            if not isinstance(self.sampler, RandomBatchSampler):
                key = "sampler"
        kwargs[key] = self.sampler
        return TorchDataLoader(self.dataset, **kwargs)
