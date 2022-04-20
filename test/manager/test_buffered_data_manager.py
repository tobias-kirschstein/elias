from collections import Iterable
from dataclasses import dataclass
from pathlib import Path
from time import sleep
from typing import Iterator, Type
from unittest import TestCase

from testfixtures import TempDirectory

from elias.config import Config
from elias.manager.buffered import BufferedDataLoader, BufferedDataManager
from elias.manager.data import BaseSampleDataManager
from elias.util.io import save_pickled, load_pickled


class SlowIterable(Iterable):
    _n_elements: int
    _delay: float
    _done: bool
    _n_elements_retrieved: int

    def __init__(self, n_elements: int = 10, delay: float = 0.1):
        self._n_elements = n_elements
        self._delay = delay
        self.reset()

    def slow_get(self, value):
        sleep(self._delay)
        return value

    def reset(self):
        self._done = False
        self._n_elements_retrieved = 0

    def is_done(self):
        return self._done

    def get_n_elements_retrieved(self):
        return self._n_elements_retrieved

    def __iter__(self) -> Iterator[int]:
        assert not self._done, "Call reset() before reusing SlowIterable"
        for i in range(self._n_elements):
            yield self.slow_get(i)
            self._n_elements_retrieved += 1
        self._done = True


@dataclass
class TestConfig(Config):
    a: int


@dataclass
class TestStatistics(Config):
    b: str


class TestDataManager(BaseSampleDataManager[int, TestConfig, TestStatistics]):

    def __init__(self, location: str):
        super(TestDataManager, self).__init__(location, "sample-$.p")
        self._n_samples_loaded = 0

    def reset_counter(self):
        self._n_samples_loaded = 0

    def get_n_samples_loaded(self):
        return self._n_samples_loaded

    def _save_sample(self, data: int, file_path: str):
        save_pickled(data, file_path)

    def _load_sample(self, file_path: str) -> int:
        sample = load_pickled(file_path)
        self._n_samples_loaded += 1
        return sample

    @classmethod
    def from_location(cls: Type['TestDataManager'], location: str) -> 'TestDataManager':
        pass


class BufferedDataManagerTest(TestCase):

    def test_buffered_data_loader(self):
        iterable = SlowIterable(n_elements=100, delay=0)
        data_loader = BufferedDataLoader(iterable)

        # Directly getting an element from an iterable only retrieves the first
        next(iter(iterable))
        self.assertFalse(iterable.is_done())
        iterable.reset()

        # Buffered data loader will retrieve all of the elements upon getting the first
        next(iter(data_loader))
        sleep(0.5)
        self.assertTrue(iterable.is_done())

    def test_buffered_data_loader_max_elements(self):
        iterable = SlowIterable(n_elements=100, delay=0)
        data_loader = BufferedDataLoader(iterable, size_load_buffer=10)

        next(iter(data_loader))
        sleep(0.5)
        self.assertEqual(iterable.get_n_elements_retrieved(), 10 + 1)  # 1 is already retrieved, 10 are buffered

    def test_buffered_data_manager(self):
        n_samples = 100
        with TempDirectory() as d:
            data_manager = TestDataManager(d.path)
            buffered_data_manager = BufferedDataManager(data_manager)

            for sample in range(n_samples):
                buffered_data_manager.save(sample)

            buffered_data_manager.shutdown()
            saved_samples = [p.name for p in Path(d.path).iterdir()]

            self.assertEqual(len(saved_samples), n_samples)
            # sample-1.p to sample-100.p should be present
            for i in range(n_samples):
                self.assertTrue(f"sample-{i + 1}.p" in saved_samples)

            # Test that wrapping of saving/loading config works
            test_config = TestConfig(2)
            buffered_data_manager.save_config(test_config)
            loaded_config = buffered_data_manager.load_config()
            self.assertEqual(loaded_config, test_config)

            # Test that wrapping of saving/loader statistics works
            test_statistics = TestStatistics("test")
            buffered_data_manager.save_stats(test_statistics)
            loaded_stats = buffered_data_manager.load_stats()
            self.assertEqual(loaded_stats, test_statistics)

            # Iterating through the dataset should yield 1-100 in order
            for idx, sample in enumerate(buffered_data_manager):
                self.assertEqual(idx, sample)

            data_manager.reset_counter()

            # Retrieving the first sample should trigger pre-loading all samples from the folder
            first_elem = next(iter(buffered_data_manager))
            sleep(0.5)
            self.assertEqual(data_manager.get_n_samples_loaded(), n_samples)

            buffered_data_manager.shutdown()

