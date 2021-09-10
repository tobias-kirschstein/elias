import random
from abc import abstractmethod, ABC
from asyncio import Event
from collections import Iterator
from math import ceil
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Iterable, TypeVar, Generic, List, Optional, Sized, Generator, Union

import numpy as np

from elias.artifact import ArtifactManager, ArtifactType
from elias.config import Config
from elias.fs import list_file_numbering
from elias.generic import get_type_var_instantiation
from elias.timing import Timing

ConfigType = TypeVar('ConfigType', bound=Config)
StatisticsType = TypeVar('StatisticsType', bound=Config)
SampleType = TypeVar('SampleType')


class IterableDataLoader(ABC):
    @abstractmethod
    def __iter__(self):
        pass


class RandomAccessDataLoader(Iterable):

    def __iter__(self):
        return (self[idx] for idx in range(len(self)))

    @abstractmethod
    def __getitem__(self, item):
        pass

    @abstractmethod
    def __len__(self):
        pass

    def view(self, indices: Union[slice, List[int]], exclude: bool = False) -> 'RandomAccessDataLoaderView':
        """
        Provides a proxy data manager that will only iterate over the given indices.
        If `exclude` is set the proxy data manager will access all elements except those specified by `indices`

        Parameters
        ----------
            indices: Indices to include/exclude in the proxy data manager
            exclude: whether the specified indices should be included or excluded

        Returns
        -------
            A proxy data manager
        """

        return RandomAccessDataLoaderView(self, indices, exclude=exclude)


class RandomAccessDataLoaderView(RandomAccessDataLoader):
    """
    Provides simple means of changing the iteration over a dataloader without copying the underlying data.
    """

    def __init__(self, dataloader: RandomAccessDataLoader, indices: Union[slice, List[int]], exclude: bool = False):
        """
        Parameters
        ----------
            dataloader: the underlying dataloader which is viewed at
            indices: the indices of the respective elements that will be used by the dataloader view. When the view
                is iterated over, the order of passed indices will be used.
        """

        self._dataloader = dataloader
        if isinstance(indices, slice):
            ranged_indices = range(len(self._dataloader))[indices]
            # TODO: DataLoader can only be iterated over once
            if exclude:
                ranged_indices = set(ranged_indices)
                self._indices = (idx for idx in range(len(self._dataloader)) if idx not in ranged_indices)
            else:
                self._indices = ranged_indices
        elif isinstance(indices, list):
            assert max(indices) < len(self._dataloader), \
                f"Cannot create view with index {max(indices)} for data loader with length {len(self._dataloader)}"
            if exclude:
                indices = set(indices)
                self._indices = [idx for idx in range(len(self._dataloader)) if idx not in indices]
            else:
                self._indices = indices
        else:
            raise ValueError(f"view indices must be slice or list not {type(indices)}")

    def __iter__(self):
        return (self._dataloader[idx] for idx in self._indices)

    def __getitem__(self, idx: int):
        return self._dataloader[self._indices[idx]]

    def __len__(self):
        return len(self._indices)


class BaseDataManager(Generic[SampleType, ConfigType, StatisticsType], ArtifactManager):
    """
    A DataManager provides access to dataset files stored on a file system.
    It is assumed that all dataset files reside in the same root folder.
    Optionally, the DataManager can process additional information of the dataset stored in JSON files:
        - ``config.json``: Contains all information necessary to reproduce the creation of the dataset
        - ``stats.json``: Contains statistics of the dataset that can only be obtained by a potentially costly iteration over the full dataset
    """

    def __init__(self, data_location: str,
                 shuffle: bool = False,
                 dataset_slice_prefix: str = None,
                 dataset_slice_suffix: str = None,
                 artifact_type: ArtifactType = ArtifactType.JSON):
        """
        Parameters
        ----------
            data_location: str
                Root folder of stored data. This is where ``config.json`` and ``stats.json`` will be loaded from.
            shuffle: bool, default False
                Only relevant when using :meth:`_lazy_load_slices`. If shuffle is set to ``True`` dataset files will
                be loaded in random order
            dataset_slice_prefix: str, optional
                Only relevant when using :meth:`_lazy_load_slices`. Dataset files are assumed to have a numbering and
                their file names follow this pattern: "``dataset_slice_prefix`` n ``dataset_slice_suffix``". Specifying
                `dataset_slice_prefix` and `dataset_slice_suffix` is necessary for :meth:`_lazy_load_slices` to be
                used and indicates how the dataset files are named
            dataset_slice_suffix: str, optional
                See explanation of `dataset_slice_prefix`

        Type Vars
        ---------
            ConfigType:
                The class of the dataset configuration (stored in ``config.json``) which is assumed to be a dataclass
                subclassing :class:`elias.config.Config`. :meth:`save_config` and :meth:`load_config` take/retrieve the
                dataset configuration as a Python object of this class.
            StatisticsType:
                Same explanation as for `ConfigType` just for the dataset statistics ``stats.json``
        """
        assert dataset_slice_prefix is None and dataset_slice_suffix is None or \
               dataset_slice_prefix is not None and dataset_slice_suffix is not None, \
            "dataset_slice_prefix and dataset_slice_suffix have to be specified together"
        super(BaseDataManager, self).__init__(data_location, artifact_type=artifact_type)

        self._data_location = data_location
        self._dataset_slice_prefix = dataset_slice_prefix
        self._dataset_slice_suffix = dataset_slice_suffix
        self._shuffle = shuffle
        self._config_cls = get_type_var_instantiation(self, ConfigType)
        self._statistics_cls = get_type_var_instantiation(self, StatisticsType)

    @staticmethod
    def to_batches(generator: Iterable, batch_size: int, lazy: bool = False) -> Generator[List[SampleType], None, None]:
        """
        Lazyly evaluated batch-wise loading
        """

        if batch_size == 1:
            for item in generator:
                yield item
            return

        if lazy:
            # Lazy returns batches as a generator where objects are only touched upon actually querying them
            iterator = iter(generator)
            try:
                while True:
                    first = next(iterator)

                    def chunk():
                        try:
                            yield first
                            for _ in range(batch_size - 1):
                                yield next(iterator)
                        except StopIteration:
                            pass

                    yield chunk()
            except StopIteration:
                pass
        else:
            # Regular mode materializes all objects within a batch before the batch is returned as a list
            batch = []
            for i, item in enumerate(generator):
                batch.append(item)
                if (i + 1) % batch_size == 0:
                    yield batch
                    batch = []
            if batch:
                yield batch

    @staticmethod
    def batchify_tensor(tensor, batch_size: int) -> Iterable:
        try:
            n_samples = len(tensor)
        except Exception:
            try:
                n_samples = tensor.shape[0]
            except Exception:
                raise ValueError(f"Cannot infer length of passed tensor with type {type(tensor)}. "
                                 f"Ensure to use a common Tensor/Array format")

        n_batches = ceil(n_samples / batch_size)
        for i_batch in range(n_batches):
            if i_batch == n_batches - 1:
                yield tensor[i_batch * batch_size:]  # Return all remaining samples as the last batch
            else:
                yield tensor[i_batch * batch_size: (i_batch + 1) * batch_size]

    def _lazy_load_slices(self) -> Iterable[str]:
        """
        Private generator for providing paths to dataset slices one by one. Implements shuffling of dataset slices.
        """

        assert self._dataset_slice_prefix is not None and self._dataset_slice_suffix is not None, \
            "_lazy_load_files() can only be used if prefix and suffix are defined"

        # Prepare all dataset batches in _data_location for lazy loading. This is necessary as loading everything at
        # once would not fit into the main memory.
        file_ids = list_file_numbering(self._data_location, self._dataset_slice_prefix, self._dataset_slice_suffix)

        if self._shuffle:
            random.shuffle(file_ids)
        if not file_ids:
            raise Exception(f"No dataset files found in {self._data_location}. Is the path correct?")

        # Converts the path list into a generator
        for file_id in file_ids:
            yield self.get_dataset_slice_path(file_id)

    def get_dataset_slice_path(self, dataset_slice_id: int) -> str:
        dataset_slice_path = f"{self._data_location}/{self._dataset_slice_prefix}{dataset_slice_id}{self._dataset_slice_suffix}"
        assert Path(dataset_slice_path).exists(), f"Could not find dataset slice `{dataset_slice_path}`"

        return dataset_slice_path

    def read(self, batch_size: int = 1) -> Iterable[SampleType]:
        return self.to_batches(self, batch_size)

    def load_config(self) -> ConfigType:
        json_config = self._load_artifact("config")
        return self._config_cls.from_json(json_config)

    def save_config(self, config: ConfigType):
        self._save_artifact(config.to_json(), "config")

    def load_stats(self) -> StatisticsType:
        json_statistics = self._load_artifact("stats")
        return self._statistics_cls.from_json(json_statistics)

    def save_stats(self, stats: StatisticsType):
        self._save_artifact(stats.to_json(), "stats")

    @abstractmethod
    def save(self, data: object, **kwargs):
        pass

    @abstractmethod
    def __iter__(self):
        pass


class IterableDataManager(Iterable, BaseDataManager[SampleType, ConfigType, StatisticsType], ABC):
    pass


class RandomAccessDataManager(RandomAccessDataLoader, BaseDataManager[SampleType, ConfigType, StatisticsType], ABC):
    pass


class CombinedIterableStopCriterion:

    @abstractmethod
    def should_stop(self, just_depleted_dl_idx: int, remaining_data_loader_indices: List[int]) -> bool:
        pass


class CombinedIterableStopCriterionAnyEmpty(CombinedIterableStopCriterion):

    def should_stop(self, just_depleted_dl_idx: int, remaining_data_loader_indices: List[int]) -> bool:
        return True


class CombinedIterableStopCriterionAllEmpty(CombinedIterableStopCriterion):

    def should_stop(self, just_depleted_dl_idx: int, remaining_data_loader_indices: List[int]) -> bool:
        return len(remaining_data_loader_indices) == 0


class CombinedIterableStopCriterionSpecificEmpty(CombinedIterableStopCriterion):

    def __init__(self, specific_dl_idx: int):
        self._specific_dl_idx = specific_dl_idx

    def should_stop(self, just_depleted_dl_idx: int, remaining_data_loader_indices: List[int]) -> bool:
        return just_depleted_dl_idx == self._specific_dl_idx


class CombinedIterableDataLoader(Iterable):
    # TODO: Add capabilities for steering how iterables are traversed when shuffle=False

    def __init__(self,
                 data_loaders: List[Iterable],
                 shuffle: bool = False,
                 sample_weights: Optional[List[float]] = None,
                 stop_criterion: CombinedIterableStopCriterion = CombinedIterableStopCriterionAllEmpty(),
                 return_dl_idx: bool = True):
        """
        Combines the specified iterables into a single iterable dataloader. Per default, the given iterables will
        be traversed in order. If :paramref:`~params.shuffle` is set, the dataloaders will be traversed randomly.

        Parameters
        ----------
            data_loaders:
                the iterables that should be combined
            shuffle:
                whether the iterables will be traversed in order or randomly
            sample_weights:
                If :paramref:`~params.shuffle` is `True`, the sample weights specify for each dataloader how likely
                it should be to draw from it. If it is `None` elements will be drawn uniformly from the given iterables
            stop_criterion:
                specifies under what circumstances the combined dataloader should stop. E.g., whether it should exhaust
                all given dataloaders, stop when any of the iterables is empty or continue until a specific dataloader
                is empty
            return_dl_idx:
                whether the returned elements of the combined dataloader should be a tuple containing the index of the
                original dataloader and the actual element (similar to Python's enumerate()).
        """

        self._data_loaders = data_loaders
        self._shuffle = shuffle
        self._return_dl_idx = return_dl_idx

        if shuffle:
            if sample_weights is None:
                # uniform random sampling
                self._sample_weights = np.array([1 / len(data_loaders) for _ in range(len(data_loaders))])
            else:
                self._sample_weights = None if sample_weights is None else np.array(sample_weights) / sum(
                    sample_weights)
                assert sample_weights is None or len(sample_weights) == len(data_loaders), \
                    f"Need to specify as many sample weights (got {len(self._sample_weights)}) " \
                    f"as dataloaders ({len(data_loaders)})"
        else:
            assert sample_weights is None, f"shuffle has to be set, if sample_weights are used"
            self._sample_weights = None

        self._stop_criterion = stop_criterion

    def save(self, data, **kwargs):
        raise Exception('CombinedDataManager cannot save')

    def __iter__(self):
        return CombinedIterableDataLoader.Iterator([iter(data_manager) for data_manager in self._data_loaders],
                                                   self._sample_weights,
                                                   self._stop_criterion,
                                                   self._return_dl_idx)

    class Iterator:

        def __init__(self, iterators: List[Iterator],
                     sample_weights: Optional[np.array],
                     stop_criterion: CombinedIterableStopCriterion,
                     return_dl_idx: bool):
            self._iterators = iterators
            self._sample_weights = sample_weights
            self._stop_criterion = stop_criterion
            self._return_dl_idx = return_dl_idx

            self._identifiers = list(range(len(iterators)))

        def __next__(self):
            if len(self._identifiers) == 0:
                raise StopIteration()

            if self._sample_weights is not None:
                sample_weights = self._sample_weights[self._identifiers]
                assert sum(sample_weights) > 0, f"sample_weights (initial: {self._sample_weights}) sum to 0"
                sample_weights /= sum(sample_weights)
                iterator_idx = np.random.choice(self._identifiers, p=sample_weights)
            else:
                # Iterate through all iterators in order
                iterator_idx = self._identifiers[0]

            try:
                sample = next(self._iterators[iterator_idx])
                if self._return_dl_idx:
                    return iterator_idx, sample
                else:
                    return sample
            except StopIteration:
                self._identifiers.remove(iterator_idx)

                if self._stop_criterion.should_stop(iterator_idx, self._identifiers):
                    raise StopIteration()
                else:
                    return next(self)


class CombinedRandomAccessDataLoader(RandomAccessDataLoader):

    def __init__(self, data_loaders: List[RandomAccessDataLoader], shuffle=False):
        # TODO: sample_weights
        self._data_loaders = data_loaders
        self._shuffle = shuffle

        if shuffle:
            self._shuffled_indices = list(range(len(self)))
            np.random.shuffle(self._shuffled_indices)

    def __iter__(self):
        return (self[idx] for idx in range(len(self)))

    def __getitem__(self, idx: int):
        assert -len(self) <= idx < len(
            self), f"Index {idx} is out of bounds for combined data loader of size {len(self)}"
        if self._shuffle:
            idx = self._shuffled_indices[idx]

        dl_idx, sample_idx = self._get_dl_idx_for_sample(idx)
        sample = self._data_loaders[dl_idx][sample_idx]

        return dl_idx, sample

    def __len__(self):
        return sum([len(data_loader) for data_loader in self._data_loaders])

    def _get_dl_idx_for_sample(self, idx: int):
        assert -len(self) <= idx < len(
            self), f"Index {idx} is out of bounds for combined data loader of size {len(self)}"
        seen_samples = 0
        current_dl_idx = 0
        if idx < 0:
            idx += len(self)
        for data_loader in self._data_loaders:
            seen_samples += len(data_loader)
            if idx < seen_samples:
                return current_dl_idx, idx - (seen_samples - len(data_loader))
            current_dl_idx += 1


# TODO: revise doc
class BufferedDataLoader(Iterable):
    """
    Wrapper class for arbitrary data managers that preloads samples in the background and provides asynchroneous saving.
    Useful in multiprocessing settings where we can have the main process preloading data while other processes do the
    work.
    The idea is to eliminate any waiting when iterating over the samples or when saving a dataset slice. This is
    obtained by using background worker threads that operate on queues instead of directly using the data_manager.
    To ensure that the python process can end after using a BufferedDataManager, one should call the .shutdown() method
    """

    QUEUE_END_MSG = 'DONE'  # A special message that is used for internal queues to signalize that the producer thread is done

    def __init__(self, data_loader: Iterable, size_load_buffer=5000):
        """
        :param data_manager: can be an arbitrary data manager that supports iterating over samples and saving dataset files
        :param size_load_buffer: specifies how many SAMPLES will be prefetched from data_manager
        """

        self._data_loader = data_loader
        self._load_buffer = Queue(size_load_buffer)
        self._load_worker = None  # Will be initialized upon obtaining an iterator
        self._stop_event = Event()

    def __iter__(self):
        """
        Initializes a worker for prefetching data. The worker will start populating the internal queue once an iterator
        is created. To avoid spawning multiple workers, one can only have one iterator at a time.
        """

        if self._load_worker is not None:
            raise Exception("There is already an iterator running!")
        self._load_worker = self.LoadWorker(self._data_loader, self._load_buffer, self._stop_event)
        self._load_worker.start()
        return BufferedDataLoader.Iterator(self)

    def __len__(self):
        try:
            if isinstance(self._data_loader, Sized):
                return len(self._data_loader)
        except TypeError:
            pass

        raise TypeError("Underlying dataloader did not specify len()")

    class Iterator(Iterator):

        def __init__(self, buffered_data_loader):
            self._buffered_data_loader: BufferedDataLoader = buffered_data_loader

        def __next__(self):
            """
            Reads from the internal buffer and only blocks when it is empty. In this case, it might help to increase the
            size of the internal buffer via size_load_buffer
            """

            print(self._buffered_data_loader._load_buffer.qsize())

            data = self._buffered_data_loader._load_buffer.get()
            if data == BufferedDataLoader.QUEUE_END_MSG:
                # the load worker will put a special DONE MESSAGE to the internal queue to signal that the data_manager
                # won't provide more samples
                self._buffered_data_loader._load_worker.join()
                self._buffered_data_loader._load_worker = None
                raise StopIteration
            return data

    def __del__(self):
        """
        Destructor. Attempts to join all threads to allow the python script to exit cleanly.
        """

        self.shutdown()

    def shutdown(self):
        """
        Clears all the buffers, terminates all workers and prepares the buffered data manager to be used again.
        Should be called when one is done with iterating over the samples to allow the python process to end.
        :return:
        """

        self._stop_event.set()  # Signalize the load worker to shutdown
        if self._load_worker:
            if self._load_worker.is_alive() and not self._load_buffer.empty():
                # In this case, the load worker is waiting to put something into the queue and thus cannot receive the
                # stop signal. Resolve by taking one element out of the read buffer
                self._load_buffer.get()
            self._load_worker.join()

        self._load_buffer.queue.clear()
        self._stop_event = Event()
        self._load_worker = None

    class LoadWorker(Thread):
        """
        Background thread that iterates over all samples in data_manager and puts them onto the internal load buffer.
        To avoid out of memory issues, the internal queue has limited size which can be controlled via size_load_buffer
        """

        def __init__(self, data_loader: Iterable, read_buffer: Queue, stop_event: Event):
            Thread.__init__(self)
            self._data_loader = data_loader
            self._read_buffer = read_buffer
            self._stop_event = stop_event

        def run(self) -> None:
            with Timing() as t:
                for sample in self._data_loader:
                    print(f"Loading sample took {t.measure(): .3f}s")

                    if self._stop_event.is_set():
                        return
                    self._read_buffer.put(sample)
                    # Signalize that the data_manager iterator is empty
                self._read_buffer.put(BufferedDataManager.QUEUE_END_MSG)


# TODO: revise BufferedDataManager
class BufferedDataManager(BaseDataManager):
    """
    Wrapper class for arbitrary data managers that preloads samples in the background and provides asynchroneous saving.
    Useful in multiprocessing settings where we can have the main process preloading data while other processes do the
    work.
    The idea is to eliminate any waiting when iterating over the samples or when saving a dataset slice. This is
    obtained by using background worker threads that operate on queues instead of directly using the data_manager.
    To ensure that the python process can end after using a BufferedDataManager, one should call the .shutdown() method
    """

    QUEUE_END_MSG = 'DONE'  # A special message that is used for internal queues to signalize that the producer thread is done

    def __init__(self, data_manager: BaseDataManager, size_load_buffer=5000, size_save_buffer=1):
        """
        :param data_manager: can be an arbitrary data manager that supports iterating over samples and saving dataset files
        :param size_load_buffer: specifies how many SAMPLES will be prefetched from data_manager
        :param size_save_buffer: specifies how many DATASET SLICES will be buffered until a call to .save() will actually block
        """
        super(BufferedDataManager, self).__init__(data_manager._data_location)

        self.data_manager = data_manager
        self.load_buffer = Queue(size_load_buffer)
        self.save_buffer = Queue(size_save_buffer)
        self.load_worker = None  # Will be initialized upon obtaining an iterator
        self.save_worker = None  # Will be initialized when the first path needs to be saved
        self.stop_event = Event()

    def __iter__(self):
        """
        Initializes a worker for prefetching data. The worker will start populating the internal queue once an iterator
        is created. To avoid spawning multiple workers, one can only have one iterator at a time.
        """

        if self.load_worker is not None:
            raise Exception("There is already an iterator running!")
        self.load_worker = self.LoadWorker(self.data_manager, self.load_buffer, self.stop_event)
        self.load_worker.start()
        return self

    def __next__(self):
        """
        Reads from the internal buffer and only blocks when it is empty. In this case, it might help to increase the
        size of the internal buffer via size_load_buffer
        """

        data = self.load_buffer.get()
        if data == self.QUEUE_END_MSG:
            # the load worker will put a special DONE MESSAGE to the internal queue to signal that the data_manager
            # won't provide more samples
            self.load_worker.join()
            self.load_worker = None
            raise StopIteration
        return data

    def __del__(self):
        """
        Destructor. Attempts to join all threads to allow the python script to exit cleanly.
        """

        self.shutdown()

    def save(self, data, **kwargs):
        """
        Puts the data on the internal save buffer and immediately returns. Only blocks when the internal save buffer
        is already full, i.e., the worker takes too longer to save one dataset slice than new data is incoming.
        In this case, this code has to be extended to allow for multiple save workers.
        A save worker is created when .save() is called for the first time.
        :param data: the data to be saved
        """

        if not self.save_worker:
            self.save_worker = self.SaveWorker(self.data_manager, self.save_buffer)
            self.save_worker.start()
        self.save_buffer.put(data)

    def shutdown(self):
        """
        Clears all the buffers, terminates all workers and prepares the buffered data manager to be used again.
        Should be called when one is done with iterating over the samples to allow the python process to end.
        :return:
        """

        self.stop_event.set()  # Signalize the load worker to shutdown
        if self.load_worker:
            if self.load_worker.is_alive() and not self.load_buffer.empty():
                # In this case, the load worker is waiting to put something into the queue and thus cannot receive the
                # stop signal. Resolve by taking one element out of the read buffer
                self.load_buffer.get()
            self.load_worker.join()

        # Possibly awake blocking SaveWorker and signalize that no more data
        # will be put to the save buffer, i.e., the worker can shutdown
        self.save_buffer.put(self.QUEUE_END_MSG)
        if self.save_worker:
            self.save_worker.join()

        self.load_buffer.queue.clear()
        self.save_buffer.queue.clear()
        self.stop_event = Event()
        self.load_worker = None
        self.save_worker = None

    class LoadWorker(Thread):
        """
        Background thread that iterates over all samples in data_manager and puts them onto the internal load buffer.
        To avoid out of memory issues, the internal queue has limited size which can be controlled via size_load_buffer
        """

        def __init__(self, data_manager, read_buffer, stop_event):
            Thread.__init__(self)
            self.data_manager = data_manager
            self.read_buffer = read_buffer
            self.stop_event = stop_event

        def run(self) -> None:
            for sample in self.data_manager.read():
                if self.stop_event.is_set():
                    return
                self.read_buffer.put(sample)
            self.read_buffer.put(BufferedDataManager.QUEUE_END_MSG)  # Signalize that the data_manager iterator is empty

    class SaveWorker(Thread):
        """
        Background thread that waits for data to be saved on the internal save buffer.
        Will run until a special DONE MESSAGE is put onto the queue.
        """

        def __init__(self, data_manager, save_buffer):
            Thread.__init__(self)
            self.data_manager = data_manager
            self.save_buffer = save_buffer

        def run(self) -> None:
            while True:
                data = self.save_buffer.get()
                if data == BufferedDataManager.QUEUE_END_MSG:
                    return
                with Timing() as t:
                    self.data_manager.save(data)

                # TODO: how to do proper logging?
                # logger.info(f"Saving {len(data)} samples took {t[0]:0.3f} seconds")
                print(f"Saving {len(data)} samples took {t[0]:0.3f} seconds")
