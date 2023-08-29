import random
import warnings
from abc import abstractmethod, ABC
from typing import Iterable, TypeVar, Generic, List, Generator, Iterator, Type, Union, Any

from numpy import deprecate
from silberstral import reveal_type_var

from elias.config import Config
from elias.data.loader import RandomAccessDataLoader
from elias.folder.folder import Folder
from elias.manager.artifact import ArtifactManager, ArtifactType
from elias.util import Version
from elias.util.batch import batchify, batchify_sliced

_ConfigType = TypeVar('_ConfigType', bound=Config)
_StatisticsType = TypeVar('_StatisticsType', bound=Config)
_SampleType = TypeVar('_SampleType')
_T = TypeVar('_T')


# TODO: We can abstract the filesystem requirement away by having an interface that provides iterating over the data
#  and querying for a list of file/data/sample/image names

class BaseDataManager(Generic[_SampleType, _ConfigType, _StatisticsType], ArtifactManager, Iterable[_SampleType], ABC):
    """
    A DataManager provides access to dataset files stored on a file system.
    It is assumed that all dataset files reside in the same root folder.
    Optionally, the DataManager can process additional information of the dataset stored in JSON files:
        - ``config.json``: Contains all information necessary to reproduce the creation of the dataset
        - ``stats.json``: Contains statistics of the dataset that can only be obtained by a potentially costly iteration over the full dataset

    Type Vars
    ---------
        _SampleType:
            The type of samples being returned by this data manager
        _ConfigType:
            The class of the dataset configuration (stored in ``config.json``) which is assumed to be a dataclass
            subclassing :class:`elias.config.Config`. :meth:`save_config` and :meth:`load_config` take/retrieve the
            dataset configuration as a Python object of this class.
        _StatisticsType:
            Same explanation as for `_ConfigType` just for the dataset statistics ``stats.json``
    """

    _data_folder: Folder
    _file_name_format: str
    _shuffle: bool
    _config_cls: _ConfigType
    _statistics_cls: _StatisticsType

    def __init__(self,
                 data_location: str,
                 run_name: str,
                 file_name_format: str,
                 shuffle: bool = False,
                 create_if_not_exists: bool = False,
                 artifact_type: ArtifactType = ArtifactType.JSON):
        """
        Parameters
        ----------
            data_location: str
                Root folder of stored data. This is where ``config.json`` and ``stats.json`` will be loaded from.
            run_name: str
                the run name or version of the stored data
            shuffle: bool, default False
                Only relevant when using :meth:`_lazy_load_slices`. If shuffle is set to ``True`` dataset files will
                be loaded in random order
            create_if_not_exists: bool, default False
                If set to True, a new folder at data_location/run_name will be created if it does not exist yet
            file_name_format:
                Format of the files in the dataset folder. As these are typically numbered, specifying a
                file name format allows convenient loading and saving of dataset files.
                An example format may be: image_$.png, sample-$.txt or dataset-$.p
        """
        super(BaseDataManager, self).__init__(f"{data_location}/{run_name}",
                                              artifact_type=artifact_type)

        self._data_folder = Folder(f"{data_location}/{run_name}", create_if_not_exists=create_if_not_exists)
        self._root_location = data_location
        self._run_name = run_name
        self._file_name_format = file_name_format
        self._shuffle = shuffle
        self._config_cls = reveal_type_var(self, _ConfigType)
        self._statistics_cls = reveal_type_var(self, _StatisticsType)

    @classmethod
    def from_location(cls: Type['BaseDataManager'],
                      location: str,
                      dataset_name: str,
                      localize_via_run_name: bool = False) -> 'BaseDataManager':
        """
        Creates a data manager for the specified location with default parameters.
        Needs to be overridden by subclasses.

        Parameters
        ----------
            location:
                path to the folder containing dataset versions
            dataset_name:
                name of the dataset linking to the folder containing preprocessed data, statistics and config files
            localize_via_run_name:
                whether only the dataset name should be used to find the folder

        Returns
        -------
            a data manager of the sub class at the specified location
        """

        try:
            if localize_via_run_name:
                data_manager = cls(dataset_name)
            else:
                data_manager = cls(location, dataset_name)
        except TypeError:
            raise NotImplementedError(f"Could not construct data manager {cls} with a single location parameter. "
                                      f"Please override from_location() to match the class __init__() method")
        return data_manager

    @staticmethod
    @deprecate
    def to_batches(generator: Iterable[_T], batch_size: int, lazy: bool = False) -> Generator[List[_T], None, None]:
        warnings.warn('This function is deprecated. Use elias.util.batch.batchify() instead', DeprecationWarning)
        return batchify(generator, batch_size, lazy=lazy)

    @staticmethod
    @deprecate
    def batchify_tensor(tensor, batch_size: int) -> Iterator:
        warnings.warn('This function is deprecated. Use elias.util.batch.batchify_slice() instead', DeprecationWarning)
        return batchify_sliced(tensor, batch_size)

    def iter_batched(self, batch_size: int) -> Iterator[_SampleType]:
        return batchify(self, batch_size)

    def load_config(self) -> _ConfigType:
        json_config = self._load_artifact("config")
        return self._config_cls.from_json(json_config)

    def save_config(self, config: _ConfigType):
        self._save_artifact(config.to_json(), "config")

    def load_stats(self) -> _StatisticsType:
        json_statistics = self._load_artifact("stats")
        return self._statistics_cls.from_json(json_statistics)

    def save_stats(self, stats: _StatisticsType):
        self._save_artifact(stats.to_json(), "stats")

    def get_file_name_by_id(self, file_id: int) -> str:
        return self._data_folder.substitute(self._file_name_format, file_id)

    def get_location(self) -> str:
        return self._data_folder.get_location()

    def get_run_name(self) -> str:
        return self._run_name

    def get_dataset_version(self) -> Version:
        run_version = self._run_name
        try:
            # For versions like v1.0-some-info, strip everything after the first dash to get the version
            idx_dash = run_version.index('-')
            run_version = run_version[:idx_dash]
        except ValueError:
            # No dash found, do nothing
            pass

        return Version(run_version)

    @abstractmethod
    def __iter__(self) -> Iterator[_SampleType]:
        pass

    @abstractmethod
    def _save(self, data: Any):
        pass


class BaseSampleDataManager(BaseDataManager[_SampleType, _ConfigType, _StatisticsType]):
    """
    Assumes that all samples lie individually in the data folder.
    """

    def save_sample(self, data: _SampleType, **kwargs):
        next_file_name = self._data_folder.generate_next_name(self._file_name_format, create_folder=False)
        self._save_sample(data, f"{self._data_folder.get_location()}/{next_file_name}")

    def load_sample(self, file_name_or_id: Union[str, int]) -> _SampleType:
        if isinstance(file_name_or_id, int):
            file_name = self._data_folder.get_file_name_by_numbering(self._file_name_format, file_name_or_id)
        else:
            file_name = file_name_or_id

        return self._load_sample(f"{self._data_folder.get_location()}/{file_name}")

    def __iter__(self) -> Iterator[_SampleType]:
        file_names = self._data_folder.list_file_numbering(self._file_name_format, return_only_file_names=True)

        if self._shuffle:
            random.shuffle(file_names)
        if not file_names:
            raise Exception(f"No dataset files found in {self._data_folder.get_location()}. Is the path correct?")

        # Converts the path list into a generator
        for file_name in file_names:
            yield self._load_sample(f"{self._data_folder.get_location()}/{file_name}")

    @abstractmethod
    def _save_sample(self, data: _SampleType, file_path: str):
        pass

    @abstractmethod
    def _load_sample(self, file_path: str) -> _SampleType:
        pass

    def _save(self, data: Any):
        self.save_sample(data)


class BaseSliceDataManager(BaseDataManager[_SampleType, _ConfigType, _StatisticsType]):
    """
    Assumes that the dataset is split into so-called "slices" that themselves contain small parts of the dataset.
    For example, the dataset may be split into several pickled files where each contains 500 samples.
    """

    def save_dataset_slice(self, dataset_slice: Iterator[_SampleType]):
        next_slice_name = self._data_folder.generate_next_name(self._file_name_format, create_folder=False)
        self._save_dataset_slice(dataset_slice, f"{self._data_folder.get_location()}/{next_slice_name}")

    def load_dataset_slice(self, slice_name_or_id: Union[str, int]) -> Iterable[_SampleType]:
        if isinstance(slice_name_or_id, int):
            slice_name = self._data_folder.get_file_name_by_numbering(self._file_name_format, slice_name_or_id)
        else:
            slice_name = slice_name_or_id

        return self._load_dataset_slice(f"{self._data_folder.get_location()}/{slice_name}")

    def __iter__(self) -> Iterator[_SampleType]:
        slice_names = self._data_folder.list_file_numbering(self._file_name_format, return_only_file_names=True)

        if self._shuffle:
            random.shuffle(slice_names)
        if not slice_names:
            raise Exception(f"No dataset files found in {self._data_folder.get_location()}. Is the path correct?")

        # Converts the path list into a generator
        for slice_name in slice_names:
            for sample in self._load_dataset_slice(f"{self._data_folder.get_location()}/{slice_name}"):
                yield sample

    @abstractmethod
    def _save_dataset_slice(self, dataset_slice: Iterator[_SampleType], slice_path: str):
        pass

    @abstractmethod
    def _load_dataset_slice(self, slice_name: str) -> Iterable[_SampleType]:
        pass

    def _save(self, data: Any):
        self.save_dataset_slice(data)


class RandomAccessSampleDataManager(RandomAccessDataLoader[_SampleType],
                                    BaseSampleDataManager[_SampleType, _ConfigType, _StatisticsType],
                                    ABC):
    pass


class RandomAccessSliceDataManager(RandomAccessDataLoader[_SampleType],
                                   BaseSliceDataManager[_SampleType, _ConfigType, _StatisticsType],
                                   ABC):
    pass
