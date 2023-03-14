import sys
import os
import re
import abc
import functools
import torch
from torch import Tensor
from torch.utils.data import Dataset as TorchDataset
import rasterio
import rasterio.features
import rasterio.merge
from rasterio.windows import from_bounds
from rasterio.io import DatasetReader
from rasterio.crs import CRS
from rasterio.vrt import WarpedVRT
from pathlib import Path
import geopandas as gpd
import numpy as np
from pandas.errors import InvalidColumnName
import warnings
import pyproj
import shapely
import shapely.geometry
import shapely.ops
from rtree.index import Index, Property
from .utils import BoundingBox, concat_samples, merge_samples, disambiguate_timestamp
from typing import (
    overload,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    cast,
    Union,
)


def raw_path(path):
    path = str(path)
    if path.startswith("~"):
        # resolve '~' to home dir
        if len(path) == 1:
            idx = 1
        elif len(path) > 1 and path[1] == "/":
            idx = 2
        else:
            raise
        path = Path.home() / path[idx:]
    # resolve path
    path = Path(path).resolve()
    return str(path)


def _get_file_generator(root, filename_glob):
    if isinstance(root, (str, Path)):
        root = raw_path(root)
        root = Path(root)
        if not root.exists():
            warnings.warn(f"root {root} does not exist")
            yield from ()
        elif root.is_dir():
            yield from root.rglob(filename_glob)
        elif root.is_file():
            yield root
        else:
            raise ValueError("'root' must be file, or directory")
    elif isinstance(root, Sequence):
        for i in root:
            yield from _get_file_generator(i, filename_glob)
    else:
        raise ValueError("'root' must be string or Path")


class Dataset(TorchDataset[Dict[str, Any]], abc.ABC):
    """Abstract base class for datasets containing geospatial information.

    Geospatial information includes things like:

    * coordinates (latitude, longitude)
    * :term:`coordinate reference system (CRS)`
    * resolution

    :class:`Dataset` is a special class of datasets. Unlike :class:`VisionDataset`,
    the presence of geospatial information allows two or more datasets to be combined
    based on latitude/longitude. This allows users to do things like:

    * Combine image and target labels and sample from both simultaneously
      (e.g. Landsat and CDL)
    * Combine datasets for multiple image sources for multimodal learning or data fusion
      (e.g. Landsat and Sentinel)

    These combinations require that all queries are present in *both* datasets,
    and can be combined using an :class:`IntersectionDataset`:

    .. code-block:: python

       dataset = landsat & cdl

    Users may also want to:

    * Combine datasets for multiple image sources and treat them as equivalent
      (e.g. Landsat 7 and Landsat 8)
    * Combine datasets for disparate geospatial locations
      (e.g. Chesapeake NY and PA)

    These combinations require that all queries are present in *at least one* dataset,
    and can be combined using a :class:`UnionDataset`:

    .. code-block:: python

       dataset = landsat7 | landsat8
    """

    #: Resolution of the dataset in units of CRS.
    res: Tuple[float, float]
    _crs: CRS

    # NOTE: according to the Python docs:
    #
    # * https://docs.python.org/3/library/exceptions.html#NotImplementedError
    #
    # the correct way to handle __add__ not being supported is to set it to None,
    # not to return NotImplemented or raise NotImplementedError. The downside of
    # this is that we have no way to explain to a user why they get an error and
    # what they should do instead (use __and__ or __or__).

    #: :class:`Dataset` addition can be ambiguous and is no longer supported.
    #: Users should instead use the intersection or union operator.
    __add__ = None  # type: ignore[assignment]

    def __init__(
        self, transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None
    ) -> None:
        """Initialize a new Dataset instance.

        Args:
            transforms: a function/transform that takes an input sample
                and returns a transformed version
        """
        self.transforms = transforms

        # Create an R-tree to index the dataset
        self.index = Index(interleaved=False, properties=Property(dimension=3))

    @abc.abstractmethod
    def __getitem__(self, query: BoundingBox) -> Dict[str, Any]:
        """Retrieve image/mask and metadata indexed by query.

        Args:
            query: (minx, maxx, miny, maxy, mint, maxt) coordinates to index

        Returns:
            sample of image/mask and metadata at that index

        Raises:
            IndexError: if query is not found in the index
        """

    def __and__(self, other: "Dataset") -> "IntersectionDataset":
        """Take the intersection of two :class:`Dataset`.

        Args:
            other: another dataset

        Returns:
            a single dataset

        Raises:
            ValueError: if other is not a :class:`Dataset`

        .. versionadded:: 0.2
        """
        return IntersectionDataset(self, other)

    def __or__(self, other: "Dataset") -> "UnionDataset":
        """Take the union of two GeoDatasets.

        Args:
            other: another dataset

        Returns:
            a single dataset

        Raises:
            ValueError: if other is not a :class:`Dataset`

        .. versionadded:: 0.2
        """
        return UnionDataset(self, other)

    def __len__(self) -> int:
        """Return the number of files in the dataset.

        Returns:
            length of the dataset
        """
        count: int = len(self.index)
        return count

    def __str__(self) -> str:
        """Return the informal string representation of the object.

        Returns:
            informal string representation
        """
        return f"""\
{self.__class__.__name__} Dataset
    type: Dataset
    bbox: {self.bounds}
    size: {len(self)}"""

    # NOTE: This hack should be removed once the following issue is fixed:
    # https://github.com/Toblerity/rtree/issues/87

    def __getstate__(
        self,
    ) -> Tuple[
        Dict[Any, Any],
        List[Tuple[int, Tuple[float, float, float, float, float, float], str]],
    ]:
        """Define how instances are pickled.

        Returns:
            the state necessary to unpickle the instance
        """
        objects = self.index.intersection(self.index.bounds, objects=True)
        tuples = [(item.id, item.bounds, item.object) for item in objects]
        return self.__dict__, tuples

    def __setstate__(
        self,
        state: Tuple[
            Dict[Any, Any],
            List[Tuple[int, Tuple[float, float, float, float, float, float], str]],
        ],
    ) -> None:
        """Define how to unpickle an instance.

        Args:
            state: the state of the instance when it was pickled
        """
        attrs, tuples = state
        self.__dict__.update(attrs)
        for item in tuples:
            self.index.insert(*item)

    @property
    def bounds(self) -> BoundingBox:
        """Bounds of the index.

        Returns:
            (minx, maxx, miny, maxy, mint, maxt) of the dataset
        """
        return BoundingBox(*self.index.bounds)

    @property
    def crs(self) -> CRS:
        """:term:`coordinate reference system (CRS)` for the dataset.

        Returns:
            the :term:`coordinate reference system (CRS)`

        .. versionadded:: 0.2
        """
        return self._crs

    @crs.setter
    def crs(self, new_crs: CRS) -> None:
        """Change the :term:`coordinate reference system (CRS)` of a Dataset.

        If ``new_crs == self.crs``, does nothing, otherwise updates the R-tree index.

        Args:
            new_crs: new :term:`coordinate reference system (CRS)`

        .. versionadded:: 0.2
        """
        if new_crs == self._crs:
            return

        new_index = Index(interleaved=False, properties=Property(dimension=3))

        project = pyproj.Transformer.from_crs(
            pyproj.CRS(str(self._crs)), pyproj.CRS(str(new_crs)), always_xy=True
        ).transform
        for hit in self.index.intersection(self.index.bounds, objects=True):
            old_minx, old_maxx, old_miny, old_maxy, mint, maxt = hit.bounds
            old_box = shapely.geometry.box(old_minx, old_miny, old_maxx, old_maxy)
            new_box = shapely.ops.transform(project, old_box)
            new_minx, new_miny, new_maxx, new_maxy = new_box.bounds
            new_bounds = (new_minx, new_maxx, new_miny, new_maxy, mint, maxt)
            new_index.insert(hit.id, new_bounds, hit.object)

        self._crs = new_crs
        self.index = new_index


class IntersectionDataset(Dataset):
    """Dataset representing the intersection of two GeoDatasets.

    This allows users to do things like:

    * Combine image and target labels and sample from both simultaneously
      (e.g. Landsat and CDL)
    * Combine datasets for multiple image sources for multimodal learning or data fusion
      (e.g. Landsat and Sentinel)

    These combinations require that all queries are present in *both* datasets,
    and can be combined using an :class:`IntersectionDataset`:

    .. code-block:: python

       dataset = landsat & cdl

    .. versionadded:: 0.2
    """

    def __init__(
        self,
        dataset1: Dataset,
        dataset2: Dataset,
        collate_fn: Callable[
            [Sequence[Dict[str, Any]]], Dict[str, Any]
        ] = concat_samples,
    ) -> None:
        """Initialize a new Dataset instance.

        Args:
            dataset1: the first dataset
            dataset2: the second dataset
            collate_fn: function used to collate samples

        Raises:
            ValueError: if either dataset is not a :class:`Dataset`
        """
        super().__init__()
        self.datasets = [dataset1, dataset2]
        self.collate_fn = collate_fn

        for ds in self.datasets:
            if not isinstance(ds, Dataset):
                raise ValueError("IntersectionDataset only supports GeoDatasets")

        self._crs = dataset1.crs
        self.res = dataset1.res

        # Force dataset2 to have the same CRS/res as dataset1
        if dataset1.crs != dataset2.crs:
            print(
                f"Converting {dataset2.__class__.__name__} CRS from "
                f"{dataset2.crs} to {dataset1.crs}"
            )
            dataset2.crs = dataset1.crs
        if dataset1.res != dataset2.res:
            print(
                f"Converting {dataset2.__class__.__name__} resolution from "
                f"{dataset2.res} to {dataset1.res}"
            )
            dataset2.res = dataset1.res

        # Merge dataset indices into a single index
        self._merge_dataset_indices()

    def _merge_dataset_indices(self) -> None:
        """Create a new R-tree out of the individual indices from two datasets."""
        i = 0
        ds1, ds2 = self.datasets
        for hit1 in ds1.index.intersection(ds1.index.bounds, objects=True):
            for hit2 in ds2.index.intersection(hit1.bounds, objects=True):
                box1 = BoundingBox(*hit1.bounds)
                box2 = BoundingBox(*hit2.bounds)
                self.index.insert(i, tuple(box1 & box2))
                i += 1

    def __getitem__(self, query: BoundingBox) -> Dict[str, Any]:
        """Retrieve image and metadata indexed by query.

        Args:
            query: (minx, maxx, miny, maxy, mint, maxt) coordinates to index

        Returns:
            sample of data/labels and metadata at that index

        Raises:
            IndexError: if query is not within bounds of the index
        """
        if not query.intersects(self.bounds):
            raise IndexError(
                f"query: {query} not found in index with bounds: {self.bounds}"
            )

        # All datasets are guaranteed to have a valid query
        samples = [ds[query] for ds in self.datasets]

        return self.collate_fn(samples)

    def __str__(self) -> str:
        """Return the informal string representation of the object.

        Returns:
            informal string representation
        """
        return f"""\
{self.__class__.__name__} Dataset
    type: IntersectionDataset
    bbox: {self.bounds}
    size: {len(self)}"""


class UnionDataset(Dataset):
    """Dataset representing the union of two GeoDatasets.

    This allows users to do things like:

    * Combine datasets for multiple image sources and treat them as equivalent
      (e.g. Landsat 7 and Landsat 8)
    * Combine datasets for disparate geospatial locations
      (e.g. Chesapeake NY and PA)

    These combinations require that all queries are present in *at least one* dataset,
    and can be combined using a :class:`UnionDataset`:

    .. code-block:: python

       dataset = landsat7 | landsat8

    .. versionadded:: 0.2
    """

    def __init__(
        self,
        dataset1: Dataset,
        dataset2: Dataset,
        collate_fn: Callable[
            [Sequence[Dict[str, Any]]], Dict[str, Any]
        ] = merge_samples,
    ) -> None:
        """Initialize a new Dataset instance.

        Args:
            dataset1: the first dataset
            dataset2: the second dataset
            collate_fn: function used to collate samples

        Raises:
            ValueError: if either dataset is not a :class:`Dataset`
        """
        super().__init__()
        self.datasets = [dataset1, dataset2]
        self.collate_fn = collate_fn

        for ds in self.datasets:
            if not isinstance(ds, Dataset):
                raise ValueError("UnionDataset only supports GeoDatasets")

        self._crs = dataset1.crs
        self.res = dataset1.res

        # Force dataset2 to have the same CRS/res as dataset1
        if dataset1.crs != dataset2.crs:
            print(
                f"Converting {dataset2.__class__.__name__} CRS from "
                f"{dataset2.crs} to {dataset1.crs}"
            )
            dataset2.crs = dataset1.crs
        if dataset1.res != dataset2.res:
            print(
                f"Converting {dataset2.__class__.__name__} resolution from "
                f"{dataset2.res} to {dataset1.res}"
            )
            dataset2.res = dataset1.res

        # Merge dataset indices into a single index
        self._merge_dataset_indices()

    def _merge_dataset_indices(self) -> None:
        """Create a new R-tree out of the individual indices from two datasets."""
        i = 0
        for ds in self.datasets:
            hits = ds.index.intersection(ds.index.bounds, objects=True)
            for hit in hits:
                self.index.insert(i, hit.bounds)
                i += 1

    def __getitem__(self, query: BoundingBox) -> Dict[str, Any]:
        """Retrieve image and metadata indexed by query.

        Args:
            query: (minx, maxx, miny, maxy, mint, maxt) coordinates to index

        Returns:
            sample of data/labels and metadata at that index

        Raises:
            IndexError: if query is not within bounds of the index
        """
        if not query.intersects(self.bounds):
            raise IndexError(
                f"query: {query} not found in index with bounds: {self.bounds}"
            )

        # Not all datasets are guaranteed to have a valid query
        samples = []
        for ds in self.datasets:
            if ds.index.intersection(tuple(query)):
                samples.append(ds[query])

        return self.collate_fn(samples)

    def __str__(self) -> str:
        """Return the informal string representation of the object.

        Returns:
            informal string representation
        """
        return f"""\
{self.__class__.__name__} Dataset
    type: UnionDataset
    bbox: {self.bounds}
    size: {len(self)}"""


class RasterDataset(Dataset):
    """Abstract base class for :class:`Dataset` stored as raster files."""

    #: Names of all available bands in the dataset
    all_bands: List[str] = []

    #: Names of RGB bands in the dataset, used for plotting
    rgb_bands: List[str] = []

    #: Color map for the dataset, used for plotting
    cmap: Dict[int, Tuple[int, int, int, int]] = {}

    def __init__(
        self,
        root: Union[str, Path] = None,
        crs: Optional[CRS] = None,
        res: Optional[Tuple[float, float]] = None,
        transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        cache: bool = True,
        bands: Union[Tuple[int, ...], List[int]] = None,
        filename_glob: str = "*.tif",
        filename_regex: str = r".*_(?P<date>\d{8})\.tif",
        date_format: str = "%Y%m%d",
        is_image: bool = True,
        dtype=torch.float,
    ) -> None:
        """Initialize a new Dataset instance.

        Args:
            root: root directory where dataset can be found
            crs: :term:`coordinate reference system (CRS)` to warp to
                (defaults to the CRS of the first file found)
            res: resolution of the dataset in units of CRS
                (defaults to the resolution of the first file found)
            transforms: a function/transform that takes an input sample
                and returns a transformed version
            cache: if True, cache file handle to speed up repeated sampling

        Raises:
            FileNotFoundError: if no files are found in ``root``
        """
        super().__init__(transforms)

        self._crs = crs
        self.res = res
        self.cache = cache
        self.bands = bands
        self.is_image = is_image
        self.dtype = dtype
        self._root = root
        self.filename_glob = filename_glob
        self.filename_regex = filename_regex
        self.date_format = date_format

        self.index = self.populate_index(
            self._root,
            self.filename_glob,
            self.filename_regex,
            self.date_format,
        )

    @property
    def root(self):
        return self._root

    @root.setter
    def root(self, value):
        self._root = value
        self.index = self.populate_index(
            self._root,
            self.filename_glob,
            self.filename_regex,
            self.date_format,
        )

    def populate_index(
        self,
        root: Union[str, Path],
        filename_glob: str = "*.tif",
        filename_regex: str = r".*_(?P<date>\d{8})\.tif",
        date_format: str = "%Y%m%d",
    ) -> Index:
        # Create an R-tree to index the dataset
        index = Index(interleaved=False, properties=Property(dimension=3))
        if root is None:
            return index
        # Populate the dataset index
        i = 0
        filename_regex = re.compile(filename_regex, re.VERBOSE)
        for filepath in _get_file_generator(root, filename_glob):
            try:
                with rasterio.open(filepath) as src:
                    # See if file has a color map
                    if len(self.cmap) == 0:
                        try:
                            self.cmap = src.colormap(1)
                        except ValueError:
                            pass
                    if self._crs is None:
                        self._crs = src.crs
                    if self.res is None:
                        self.res = src.res

                    with WarpedVRT(src, crs=self._crs) as vrt:
                        minx, miny, maxx, maxy = vrt.bounds
            except rasterio.errors.RasterioIOError:
                # Skip files that rasterio is unable to read
                continue
            else:
                mint: float = 0
                maxt: float = sys.maxsize
                match = re.match(filename_regex, filepath.name)
                if match is not None and "date" in match.groupdict():
                    date = match.group("date")
                    mint, maxt = disambiguate_timestamp(date, date_format)

                coords = (minx, maxx, miny, maxy, mint, maxt)
                index.insert(i, coords, str(filepath))
                i += 1

        if i == 0:
            warnings.warn(f"No data was found in '{root}'")

        return index

    def __getitem__(self, query: BoundingBox) -> Dict[str, Any]:
        """Retrieve image/mask and metadata indexed by query.

        Args:
            query: (minx, maxx, miny, maxy, mint, maxt) coordinates to index

        Returns:
            sample of image/mask and metadata at that index

        Raises:
            IndexError: if query is not found in the index
        """
        hits = self.index.intersection(tuple(query), objects=True)
        filepaths = [hit.object for hit in hits]

        if not filepaths:
            raise IndexError(
                f"query: {query} not found in index with bounds: {self.bounds}"
            )

        data = self._merge_files(filepaths, query)

        if self.bands is not None:
            data = data[self.bands, ...]

        data = data.to(self.dtype)

        key = "image" if self.is_image else "mask"
        sample = {key: data, "crs": self.crs, "bbox": query}

        if self.transforms is not None:
            while len(sample[key].shape) < 3:
                sample[key] = sample[key].unsqueeze(0)
            sample = self.transforms(sample)
            sample[key] = sample[key].squeeze()

        return sample

    def _merge_files(self, filepaths: Sequence[str], query: BoundingBox) -> Tensor:
        """Load and merge one or more files.

        Args:
            filepaths: one or more files to load and merge
            query: (minx, maxx, miny, maxy, mint, maxt) coordinates to index

        Returns:
            image/mask at that index
        """
        if self.cache:
            vrt_fhs = [self._cached_load_warp_file(fp) for fp in filepaths]
        else:
            vrt_fhs = [self._load_warp_file(fp) for fp in filepaths]

        bounds = (query.minx, query.miny, query.maxx, query.maxy)
        if len(vrt_fhs) == 1:
            src = vrt_fhs[0]
            out_width = int(round((query.maxx - query.minx) / self.res[0]))
            out_height = int(round((query.maxy - query.miny) / self.res[1]))
            out_shape = (src.count, out_height, out_width)
            dest = src.read(
                out_shape=out_shape, window=from_bounds(*bounds, src.transform)
            )
        else:
            dest, _ = rasterio.merge.merge(vrt_fhs, bounds, self.res)

        # fix numpy dtypes which are not supported by pytorch tensors
        if dest.dtype == np.uint16:
            dest = dest.astype(np.int32)
        elif dest.dtype == np.uint32:
            dest = dest.astype(np.int64)

        tensor = torch.tensor(dest)
        return tensor

    @functools.lru_cache(maxsize=128)
    def _cached_load_warp_file(self, filepath: str) -> DatasetReader:
        """Cached version of :meth:`_load_warp_file`.

        Args:
            filepath: file to load and warp

        Returns:
            file handle of warped VRT
        """
        return self._load_warp_file(filepath)

    def _load_warp_file(self, filepath: str) -> DatasetReader:
        """Load and warp a file to the correct CRS and resolution.

        Args:
            filepath: file to load and warp

        Returns:
            file handle of warped VRT
        """
        src = rasterio.open(filepath)

        # Only warp if necessary
        if src.crs != self.crs:
            vrt = WarpedVRT(src, crs=self.crs)
            src.close()
            return vrt
        else:
            return src


class JSONDataset(Dataset):
    """Abstract base class for :class:`Dataset` stored as GeoJSON files."""

    def __init__(
        self,
        root: Union[str, Path, Sequence[str], Sequence[Path]] = "data",
        crs: CRS = None,
        res: Tuple[float, float] = (0.0001, 0.0001),
        transforms: Callable[[Dict[str, Any]], Dict[str, Any]] = None,
        exdata: List[int] = None,
        indata: List[int] = None,
        filename_glob: str = "*.geojson",
        filename_regex: str = r".*_(?P<date>\d{8})\.geojson",
        date_format: str = "%Y%m%d",
        label_column: str = None,
        dtype=torch.long,
    ) -> None:
        """AI is creating summary for __init__

        Parameters
        ----------
        root : str, optional
            Root directory where dataset can be found, by default "data"
        crs : CRS, optional
            :term:`coordinate reference system (CRS)` to warp to (defaults to
            the CRS of the first file found), by default None
        res : float, optional
            Resolution of the dataset in units of CRS, by default 0.0001
        transforms : Callable[[Dict[str, Any]], Dict[str, Any]], optional
            Entry and returns a transformed version, by default None

        Raises
        ------
        FileNotFoundError
            FileNotFoundError: if no files are found in ``root``
        """
        super().__init__(transforms)

        self._crs = crs
        self.res: Tuple[float, float] = res
        self.exdata = exdata
        self.indata = indata
        self.label_column = label_column
        self.dtype = dtype
        self._root = root
        self.filename_glob = filename_glob
        self.filename_regex = filename_regex
        self.date_format = date_format

        self.index = self.populate_index(
            self._root,
            self.filename_glob,
            self.filename_regex,
            self.date_format,
        )

    def populate_index(
        self,
        root: Union[str, Path, Sequence[str], Sequence[Path]] = "data",
        filename_glob: str = "*.geojson",
        filename_regex: str = r".*_(?P<date>\d{8})\.geojson",
        date_format: str = "%Y%m%d",
    ) -> Index:
        # Create an R-tree to index the dataset
        index = Index(interleaved=False, properties=Property(dimension=3))
        if root is None:
            return index
        # Populate the dataset index
        i = 0
        filename_regex = re.compile(filename_regex, re.VERBOSE)
        for filepath in _get_file_generator(root, filename_glob):
            try:
                # Read in vector
                gdf = gpd.read_file(filepath)
                if self._crs is None:
                    self._crs = gdf.crs
                gdf = gdf.set_crs(self._crs, allow_override=True)
            except Exception as e:
                print(e)
                # Skip files that rasterio is unable to read
                continue
            else:
                mint: float = 0
                maxt: float = sys.maxsize

                match = re.match(filename_regex, os.path.basename(filepath))
                if match is not None and "date" in match.groupdict():
                    date = match.group("date")
                    mint, maxt = disambiguate_timestamp(date, date_format)

                minx, miny, maxx, maxy = gdf.total_bounds
                coords = (minx, maxx, miny, maxy, mint, maxt)
                index.insert(i, coords, filepath)
                i += 1

        if i == 0:
            warnings.warn(f"No data was found in '{root}'")

        return index

    @property
    def root(self):
        return self._root

    @root.setter
    def root(self, value):
        self._root = value
        self.index = self.populate_index(
            self._root,
            self.filename_glob,
            self.filename_regex,
            self.date_format,
        )

    def __getitem__(self, query: BoundingBox) -> Dict[str, Any]:
        """Retrieve image/mask and metadata indexed by query.

        Parameters
        ----------
        query : BoundingBox
            (minx, maxx, miny, maxy, mint, maxt) coordinates to index

        Returns
        -------
        Dict[str, Any]
            Sample of image/mask and metadata at that index

        Raises
        ------
        IndexError
            If query is not found in the index
        """
        hits = self.index.intersection(tuple(query), objects=True)
        filepaths = [hit.object for hit in hits]

        if not filepaths:
            raise IndexError(
                f"query: {query} not found in index with bounds: {self.bounds}"
            )

        shapes = []
        for filepath in filepaths:
            # Read in vector
            gdf = gpd.read_file(filepath)
            # Get list of geometries for all features in vector file
            if self.label_column not in gdf.columns:
                msg = (
                    f"Label column '{self.label_column}' not found in "
                    "DataFrame. A valid label column must be provided where "
                    "the row entries are one integer with a value between 0 "
                    "and 255. These integers represent the class and are "
                    "used to create a raster mask for training."
                )
                raise InvalidColumnName(msg)
            for geom, label in zip(gdf.geometry, gdf[self.label_column]):
                if self.exdata is not None and label in self.exdata:
                    continue
                if self.indata is not None and label not in self.indata:
                    continue
                shapes += [(geom, int(label))]
            # shapes += [
            #     (geom, int(label)) for geom, label in
            #     zip(gdf.geometry, gdf[self.label_column])
            # ]

        # Rasterize geometries
        width = (query.maxx - query.minx) / self.res[0]
        height = (query.maxy - query.miny) / self.res[1]
        transform = rasterio.transform.from_bounds(
            query.minx, query.miny, query.maxx, query.maxy, width, height
        )
        if shapes:
            # create raster mask from labels
            masks = rasterio.features.rasterize(
                shapes,
                out_shape=(int(height), int(width)),
                transform=transform,
                dtype=np.uint8,
            )
        else:
            # If no features are found in this query, return an empty mask
            # with the default fill value and dtype used by rasterize
            masks = np.zeros((int(height), int(width)), dtype=np.uint8)

        sample = {
            "mask": torch.tensor(masks).to(self.dtype),
            "crs": self.crs,
            "bbox": query,
        }

        if self.transforms is not None:
            while len(sample["mask"].shape) < 3:
                sample["mask"] = sample["mask"].unsqueeze(0)
            sample = self.transforms(sample)
            sample["mask"] = sample["mask"].squeeze()

        return sample


class TaskDataset(Dataset):
    @overload
    def __init__(self, *args, transforms=None):
        ...

    @overload
    def __init__(self, dataset=None, transforms=None):
        ...

    @overload
    def __init__(self, image_dataset=None, label_dataset=None, transforms=None):
        ...

    def __init__(
        self,
        *args,
        dataset=None,
        image_dataset=None,
        label_dataset=None,
        transforms=None,
    ):
        super().__init__()
        # combine image and label data
        if len(args) > 0:
            self.dataset = args[0]
            for i in args[1:]:
                self.dataset &= i
        elif dataset is not None:
            if isinstance(dataset, (list, tuple)):
                self.dataset = dataset[0]
                for i in dataset[1:]:
                    self.dataset &= i
            else:
                self.dataset = dataset
        elif image_dataset is not None and label_dataset is not None:
            self.dataset = image_dataset & label_dataset
        else:
            raise
        #
        self.transforms = transforms
        # reference dataset parameters
        self.index = self.dataset.index
        self._crs = self.dataset.crs
        self.res: Tuple[float, float] = self.dataset.res

    def __getitem__(
        self,
        query: BoundingBox,
    ) -> Dict[str, Any]:
        batch = self.dataset[query]
        # transform
        if self.transforms is not None:
            # image and mask must have same dimension (C x H x W)
            if len(batch["mask"].shape) < len(batch["image"].shape):
                batch["mask"] = batch["mask"].unsqueeze(0)
            batch = self.transforms(batch)
            # transform returns 4 dimensional tensor (B x C x H x W)
            batch["image"] = batch["image"].squeeze()
            batch["mask"] = batch["mask"].squeeze()
        # cannot return 'bbox'---which is a frozen dataclass---but lightning
        # tries to modify all outputs
        out = {
            "image": batch["image"],
            "mask": batch["mask"],
        }
        if "files" in batch.keys():
            out["files"] = batch["files"]
        return out
