import abc
import random
from rtree.index import Index, Property
from torch.utils.data import Sampler as TorchSampler
from .datasets import Dataset
from .utils import BoundingBox, to_tuple, get_random_bounding_box
import numpy as np
from typing import List, Optional, Tuple, Union, Iterator, Sized


class Sampler(TorchSampler[List[BoundingBox]], abc.ABC):
    """Abstract base class for sampling from :class:`~torchgeo.datasets.Dataset`.

    Unlike PyTorch's :class:`~torch.utils.data.BatchSampler`, :class:`BatchGeoSampler`
    returns enough geospatial information to uniquely index any
    :class:`~torchgeo.datasets.Dataset`. This includes things like latitude,
    longitude, height, width, projection, coordinate system, and time.
    """

    def __init__(
        self,
        size: Union[float, Tuple[float, float]] = None,
        stride: Union[Tuple[float, float], float] = None,
        length: int = None,
        batch_size: int = None,
        dataset: Optional[Dataset] = None,
        data_source: Optional[Sized] = None,
        roi: Optional[BoundingBox] = None,
    ) -> None:
        """Initialize a new Sampler instance.

        Args:
            dataset: dataset to index from
            data_source:
            roi: region of interest to sample from (minx, maxx, miny, maxy, mint, maxt)
                (defaults to the bounds of ``dataset.index``)
        """
        super().__init__(data_source)

        self.size = to_tuple(size)
        self.stride = to_tuple(stride)
        self.length = length
        self.batch_size = batch_size

        self._dataset = dataset
        self.roi = roi

        self.index = None
        self.res = None
        self.hits = None

        if dataset is not None:
            self.init_dataset(dataset, roi)

    @property
    def dataset(self):
        return self._dataset

    @dataset.setter
    def dataset(self, value):
        self._dataset = value
        self.init_dataset(value)

    def init_dataset(self, dataset, roi: Optional[BoundingBox] = None):
        if roi is None and self.roi is None:
            self.index = dataset.index
            self.roi = BoundingBox(*self.index.bounds)
        else:
            if roi is not None:
                self.roi = roi
            self.index = Index(interleaved=False, properties=Property(dimension=3))
            hits = dataset.index.intersection(tuple(roi), objects=True)
            for hit in hits:
                bbox = BoundingBox(*hit.bounds) & self.roi
                self.index.insert(hit.id, tuple(bbox), hit.object)

        self.res = dataset.res
        if self.size is not None:
            self.size = (
                self.size[0] * self.res[1],
                self.size[1] * self.res[0],
            )
        if self.stride is not None:
            self.stride = (
                self.stride[0] * self.res[1],
                self.stride[1] * self.res[0],
            )

        self.hits = []
        for hit in self.index.intersection(tuple(self.roi), objects=True):
            bounds = BoundingBox(*hit.bounds)
            if (
                bounds.maxx - bounds.minx > self.size[1]
                and bounds.maxy - bounds.miny > self.size[0]
            ):
                self.hits.append(hit)

        if self.length is None:
            length: int = 0
            for hit in self.hits:
                bounds = BoundingBox(*hit.bounds)

                rows = (
                    int((bounds.maxy - bounds.miny - self.size[0]) // self.stride[0])
                    + 1
                )
                cols = (
                    int((bounds.maxx - bounds.minx - self.size[1]) // self.stride[1])
                    + 1
                )
                length += rows * cols
            self.length = length

    @abc.abstractmethod
    def __iter__(self) -> Iterator[Union[BoundingBox, List[BoundingBox]]]:
        """Return a batch of indices of a dataset.

        Returns:
            batch of (minx, maxx, miny, maxy, mint, maxt) coordinates to index a dataset
        """


class RandomBatchSampler(Sampler):
    """Samples batches of elements from a region of interest randomly.

    This is particularly useful during training when you want to maximize the size of
    the dataset and return as many random :term:`chips <chip>` as possible.
    """

    def __init__(
        self,
        size: Union[Tuple[float, float], float],
        batch_size: int,
        length: int,
        dataset: Dataset = None,
        roi: Optional[BoundingBox] = None,
    ) -> None:
        """Initialize a new Sampler instance.

        The ``size`` argument can either be:

        * a single ``float`` - in which case the same value is used for the height and
          width dimension
        * a ``tuple`` of two floats - in which case, the first *float* is used for the
          height dimension, and the second *float* for the width dimension

        Args:
            dataset: dataset to index from
            size: dimensions of each :term:`patch` in units of CRS
            batch_size: number of samples per batch
            length: number of samples per epoch
            roi: region of interest to sample from (minx, maxx, miny, maxy, mint, maxt)
                (defaults to the bounds of ``dataset.index``)
        """
        super().__init__(
            dataset=dataset, size=size, batch_size=batch_size, length=length, roi=roi
        )

    def __iter__(self) -> Iterator[List[BoundingBox]]:
        """Return the indices of a dataset.

        Returns:
            batch of (minx, maxx, miny, maxy, mint, maxt) coordinates to index a dataset
        """
        for _ in range(len(self)):
            # Choose a random tile
            hit = random.choice(self.hits)
            bounds = BoundingBox(*hit.bounds)

            # Choose random indices within that tile
            batch = []
            for _ in range(self.batch_size):
                bounding_box = get_random_bounding_box(bounds, self.size, self.res)
                batch.append(bounding_box)

            yield batch

    def __len__(self) -> int:
        """Return the number of batches in a single epoch.

        Returns:
            number of batches in an epoch
        """
        return self.length // self.batch_size


class RandomSampler(Sampler):
    """Samples elements from a region of interest randomly.

    This is particularly useful during training when you want to maximize the size of
    the dataset and return as many random :term:`chips <chip>` as possible.

    This sampler is not recommended for use with tile-based datasets. Use
    :class:`RandomBatchSampler` instead.
    """

    def __init__(
        self,
        size: Union[Tuple[float, float], float],
        length: int,
        dataset: Dataset = None,
        roi: Optional[BoundingBox] = None,
    ) -> None:
        """Initialize a new Sampler instance.

        The ``size`` argument can either be:

        * a single ``float`` - in which case the same value is used for the height and
          width dimension
        * a ``tuple`` of two floats - in which case, the first *float* is used for the
          height dimension, and the second *float* for the width dimension

        Args:
            dataset: dataset to index from
            size: dimensions of each :term:`patch` in units of CRS
            length: number of random samples to draw per epoch
            roi: region of interest to sample from (minx, maxx, miny, maxy, mint, maxt)
                (defaults to the bounds of ``dataset.index``)
        """
        super().__init__(dataset=dataset, size=size, length=length, roi=roi)

    def __iter__(self) -> Iterator[BoundingBox]:
        """Return the index of a dataset.

        Returns:
            (minx, maxx, miny, maxy, mint, maxt) coordinates to index a dataset
        """
        for _ in range(len(self)):
            # Choose a random tile
            hit = random.choice(self.hits)
            bounds = BoundingBox(*hit.bounds)

            # Choose a random index within that tile
            bounding_box = get_random_bounding_box(bounds, self.size, self.res)

            yield bounding_box

    def __len__(self) -> int:
        """Return the number of samples in a single epoch.

        Returns:
            length of the epoch
        """
        return self.length


class FullGridSampler(Sampler):
    """Samples elements in a grid-like fashion.

    This is particularly useful during evaluation when you want to make predictions for
    an entire region of interest. You want to minimize the amount of redundant
    computation by minimizing overlap between :term:`chips <chip>`.

    Usually the stride should be slightly smaller than the chip size such that each chip
    has some small overlap with surrounding chips. This is used to prevent `stitching
    artifacts <https://arxiv.org/abs/1805.12219>`_ when combining each prediction patch.
    The overlap between each chip (``chip_size - stride``) should be approximately equal
    to the `receptive field <https://distill.pub/2019/computing-receptive-fields/>`_ of
    the CNN.
    """

    def __init__(
        self,
        size: Union[Tuple[float, float], float],
        stride: Union[Tuple[float, float], float],
        dataset: Dataset = None,
        roi: Optional[BoundingBox] = None,
    ) -> None:
        """Initialize a new Sampler instance.

        The ``size`` and ``stride`` arguments can either be:

        * a single ``float`` - in which case the same value is used for the
        height and width dimension
        * a ``tuple`` of two floats - in which case, the first *float* is used
        for the height dimension, and the second *float* for the width dimension

        Args:
            dataset: dataset to index from
            size: dimensions of each :term:`patch` in units of CRS
            stride: distance to skip between each patch
            roi: region of interest to sample from (minx, maxx, miny, maxy, mint, maxt)
                (defaults to the bounds of ``dataset.index``)
        """
        super().__init__(dataset=dataset, size=size, stride=stride, roi=roi)

    def __len__(self) -> int:
        """Return the number of samples over the ROI.

        Returns:
            number of patches that will be sampled
        """
        return self.length

    def __iter__(self) -> Iterator[BoundingBox]:
        """Return the index of a dataset.

        Automatically adjust the stride value so that the whole image is
        sampled.

        Returns:
            (minx, maxx, miny, maxy, mint, maxt) coordinates to index a dataset
        """
        for hit in self.hits:
            bounds = BoundingBox(*hit.bounds)

            ny = int((bounds.maxy - bounds.miny - self.size[0]) // self.stride[0] + 1)
            steps_y = np.linspace(
                bounds.miny,
                bounds.maxy - self.size[0],
                ny + 1,
            )

            nx = int((bounds.maxx - bounds.minx - self.size[1]) // self.stride[1] + 1)
            steps_x = np.linspace(
                bounds.minx,
                bounds.maxx - self.size[1],
                nx + 1,
            )

            mint = bounds.mint
            maxt = bounds.maxt

            for y in steps_y:
                for x in steps_x:
                    yield BoundingBox(
                        x, x + self.size[1], y, y + self.size[0], mint, maxt
                    )
