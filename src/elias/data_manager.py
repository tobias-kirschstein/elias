import random
from abc import abstractmethod
from asyncio import Event
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Iterable, TypeVar, Generic, Type

from elias.config import Config
from elias.fs import list_file_numbering
from elias.io import load_json, save_json
from elias.timing import Timing

ConfigType = TypeVar('ConfigType')
StatisticsType = TypeVar('StatisticsType')


class BaseDataManager(Generic[ConfigType, StatisticsType]):
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
                 config_cls: Type[Config] = Config,
                 statistics_cls: Type[Config] = Config):
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
            config_cls: Type[Config], default Config
                The class of the dataset configuration (stored in ``config.json``) which is assumed to be a dataclass
                subclassing :class:`elias.config.Config`. :meth:`save_config` and :meth:`load_config` take/retrieve the
                dataset configuration as a Python object of this class.
            statistics_cls: Type[Config], default Config
                Same explanation as for `config_cls` just for the dataset statistics ``stats.json``
        """

        assert Path(data_location).is_dir(), f"Specified data location '{data_location}' is not a directory"

        self._data_location = data_location
        self._dataset_slice_prefix = dataset_slice_prefix
        self._dataset_slice_suffix = dataset_slice_suffix
        self._shuffle = shuffle
        self._config_cls = config_cls
        self._statistics_cls = statistics_cls

    @staticmethod
    def to_batches(generator: Iterable, batch_size: int, lazy: bool = False) -> Iterable:
        """
        Lazyly evaluated batch-wise loading of the code snippets.
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
        return f"{self._data_location}/{self._dataset_slice_prefix}{dataset_slice_id}{self._dataset_slice_suffix}"

    def read(self, batch_size: int = 1) -> Iterable:
        return self.to_batches(self, batch_size)

    def load_config(self) -> ConfigType:
        json_config = load_json(f"{self._data_location}/config.json")
        return self._config_cls.from_json(json_config)

    def save_config(self, config: ConfigType):
        save_json(config.to_json(), f"{self._data_location}/config.json")

    def load_stats(self) -> StatisticsType:
        json_statistics = load_json(f"{self._data_location}/stats.json")
        return self._statistics_cls.from_json(json_statistics)

    def save_stats(self, stats: StatisticsType):
        save_json(stats.to_json(), f"{self._data_location}/stats.json")

    @abstractmethod
    def save(self, data: object, **kwargs):
        pass

    @abstractmethod
    def __iter__(self):
        pass


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
