import logging
from asyncio import Event
from queue import Queue
from threading import Thread
from typing import Iterable, Iterator, Sized, TypeVar, Optional, Type, Any

from elias.config import Config
from elias.manager.data import BaseDataManager
from elias.util.timing import Timing

_SampleType = TypeVar('_SampleType')

# A special message that is used for internal queues to signalize that the producer thread is done
_QUEUE_END_MSG = object()


class BufferedDataLoader(Iterable[_SampleType]):
    """
    Wrapper class for arbitrary data managers that preloads samples in the background and provides asynchroneous saving.
    Useful in multiprocessing settings where we can have the main process preloading data while other processes do the
    work.
    The idea is to eliminate any waiting when iterating over the samples or when saving a dataset slice. This is
    obtained by using background worker threads that operate on queues instead of directly using the data_manager.
    To ensure that the python process can end after using a BufferedDataManager, one should call the .shutdown() method
    """

    _data_loader: Iterable[_SampleType]
    _load_buffer: Queue
    _load_worker: Optional[Thread]
    _stop_event: Event

    def __init__(self, data_loader: Iterable[_SampleType], size_load_buffer: int = 5000):
        """

        Parameters
        ----------
            data_loader:
                can be any iterable that provides samples
            size_load_buffer:
                specifies how many samples will be prefetched from the `data_loader`
        """

        self._data_loader = data_loader
        self._load_buffer = Queue(size_load_buffer)
        self._load_worker = None  # Will be initialized upon obtaining an iterator
        self._stop_event = Event()

    def __iter__(self) -> Iterator[_SampleType]:
        """
        Initializes a worker for prefetching data. The worker will start populating the internal queue once an iterator
        is created. To avoid spawning multiple workers, one can only have one iterator at a time.
        """

        if self._load_worker is not None:
            raise Exception("There is already an iterator running!")
        self._load_worker = self.LoadWorker(self._data_loader, self._load_buffer, self._stop_event)
        self._load_worker.start()
        return BufferedDataLoader.Iterator(self)

    def __len__(self) -> int:
        try:
            if isinstance(self._data_loader, Sized):
                return len(self._data_loader)
        except TypeError:
            pass

        raise TypeError("Underlying dataloader did not specify len()")

    def __del__(self):
        """
        Destructor. Attempts to join all threads to allow the python script to exit cleanly.
        """

        self.shutdown()

    def shutdown(self):
        """
        Clears all the buffers, terminates all workers and prepares the buffered data manager to be used again.
        Should be called when one is done with iterating over the samples to allow the python process to end.
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

    # -------------------------------------------------------------------------
    # Inner classes
    # -------------------------------------------------------------------------

    class Iterator(Iterator[_SampleType]):

        _buffered_data_loader: 'BufferedDataLoader'

        def __init__(self, buffered_data_loader: 'BufferedDataLoader'):
            self._buffered_data_loader = buffered_data_loader

        def __next__(self) -> _SampleType:
            """
            Reads from the internal buffer and only blocks when it is empty. In this case, it might help to increase the
            size of the internal buffer via size_load_buffer
            """

            data = self._buffered_data_loader._load_buffer.get()
            if data == _QUEUE_END_MSG:
                # the load worker will put a special DONE MESSAGE to the internal queue to signal that the data_manager
                # won't provide more samples
                self._buffered_data_loader._load_worker.join()
                self._buffered_data_loader._load_worker = None
                raise StopIteration
            return data

    class LoadWorker(Thread):
        """
        Background thread that iterates over all samples in data_manager and puts them onto the internal load buffer.
        To avoid out of memory issues, the internal queue has limited size which can be controlled via size_load_buffer
        """

        _data_loader: Iterable[_SampleType]
        _read_buffer: Queue
        _stop_event: Event

        def __init__(self, data_loader: Iterable[_SampleType], read_buffer: Queue, stop_event: Event):
            Thread.__init__(self)
            self._data_loader = data_loader
            self._read_buffer = read_buffer
            self._stop_event = stop_event

        def run(self) -> None:
            with Timing() as t:
                for sample in self._data_loader:
                    logging.debug(f"Loading sample took {t.measure(): .3f}s")

                    if self._stop_event.is_set():
                        return
                    self._read_buffer.put(sample)
                    # Signalize that the data_manager iterator is empty
                self._read_buffer.put(_QUEUE_END_MSG)


class BufferedDataManager(BaseDataManager[_SampleType, Config, Config]):
    """
    Wrapper class for arbitrary data managers that preloads samples in the background and provides asynchroneous saving.
    Useful in multiprocessing settings where we can have the main process preloading data while other processes do the
    work.
    The idea is to eliminate any waiting when iterating over the samples or when saving a dataset slice. This is
    obtained by using background worker threads that operate on queues instead of directly using the data_manager.
    To ensure that the python process can end after using a BufferedDataManager, one should call the .shutdown() method
    """

    _data_manager: BaseDataManager
    _buffered_data_loader: BufferedDataLoader
    _save_buffer: Queue
    _save_worker: Optional[Thread]
    _stop_event: Event

    def __init__(self, data_manager: BaseDataManager, size_load_buffer: int = 5000, size_save_buffer: int = 1):
        """

        Parameters
        ----------
            data_manager:
                can be an arbitrary data manager that supports iterating over samples and saving dataset files
            size_load_buffer:
                specifies how many SAMPLES will be prefetched from data_manager
            size_save_buffer:
                specifies how many DATASET SLICES will be buffered until a call to .save() will actually block
        """

        super(BufferedDataManager, self).__init__(data_manager._root_location,
                                                  data_manager._run_name,
                                                  data_manager._file_name_format,
                                                  shuffle=data_manager._shuffle,
                                                  artifact_type=data_manager._artifact_type)

        self._data_manager = data_manager
        self._buffered_data_loader = BufferedDataLoader(data_manager, size_load_buffer=size_load_buffer)
        self._save_buffer = Queue(size_save_buffer)
        self._save_worker = None  # Will be initialized when the first path needs to be saved

        # Retrieve classes for Data config and statistics from the provided data manager
        # This is the reason why _ConfigType and _StatisticsType are set to None in the inheritance
        # of BufferedDataManager. The BufferedDataManager essentially wraps and mimics the provided data manager
        self._config_cls = data_manager._config_cls
        self._statistics_cls = data_manager._statistics_cls

    @classmethod
    def from_location(cls: Type['BufferedDataManager'], location: str) -> 'BufferedDataManager':
        raise NotImplementedError("Default BufferedDataManager does not provide initialization with from_location()")

    def __iter__(self) -> Iterator[_SampleType]:
        """
        Initializes a worker for prefetching data. The worker will start populating the internal queue once an iterator
        is created. To avoid spawning multiple workers, one can only have one iterator at a time.
        """

        return iter(self._buffered_data_loader)

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

        Parameters
        ----------
            data:
                the data to be saved
            kwargs:
                only there to allow subclasses to override the save() method
        """

        if not self._save_worker:
            self._save_worker = self.SaveWorker(self._data_manager, self._save_buffer)
            self._save_worker.start()
        self._save_buffer.put(data)

    def shutdown(self):
        """
        Waits for load and save worker to finish their tasks.
        Clears all the buffers, terminates all workers and prepares the buffered data manager to be used again.
        Should be called when one is done with iterating over the samples to allow the python process to end.
        """

        self._buffered_data_loader.shutdown()

        # Possibly awake blocking SaveWorker and signalize that no more data
        # will be put to the save buffer, i.e., the worker can shutdown
        self._save_buffer.put(_QUEUE_END_MSG)
        if self._save_worker:
            self._save_worker.join()

        self._save_buffer.queue.clear()
        self._save_worker = None

    class SaveWorker(Thread):
        """
        Background thread that waits for data to be saved on the internal save buffer.
        Will run until a special DONE MESSAGE is put onto the queue.
        """

        _data_manager: BaseDataManager
        _save_buffer: Queue

        def __init__(self, data_manager: BaseDataManager, save_buffer: Queue):
            Thread.__init__(self)
            self._data_manager = data_manager
            self._save_buffer = save_buffer

        def run(self) -> None:
            while True:
                data = self._save_buffer.get()
                if data == _QUEUE_END_MSG:
                    return
                with Timing() as t:
                    self._data_manager._save(data)

                try:
                    logging.info(f"Saving {len(data)} samples took {t[0]:0.3f} seconds")
                except TypeError:
                    # If data does not have length, don't log anything
                    pass

    def _save(self, data: Any):
        self.save(data)
