from typing import Iterable, List, Optional, Iterator, TypeVar

import numpy as np

from elias.data.loader import RandomAccessDataLoader
from elias.data.sampling import SamplingStrategy, CyclicSamplingStrategy, RandomSamplingStrategy, \
    WeightedSamplingStrategy, SequentialSamplingStrategy, ChoiceSampler
from elias.data.stop_criterion import CombinedIterableStopCriterion, CombinedIterableStopCriterionAllEmpty

_T = TypeVar('_T')


class CombinedIterableDataLoader(Iterable[_T]):
    def __init__(self,
                 data_loaders: List[Iterable[_T]],
                 shuffle: bool = False,
                 sample_weights: Optional[List[float]] = None,
                 stop_criterion: CombinedIterableStopCriterion = CombinedIterableStopCriterionAllEmpty(),
                 alternating_sampling: bool = False,
                 sampling_strategy: Optional[SamplingStrategy] = None,
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
            alternating_sampling:
                Overrides `sample_weights`. If set to `True`, provided iterators will be traversed in a round robin
                style instead of exhausting the first before continuing with the second.
            return_dl_idx:
                whether the returned elements of the combined dataloader should be a tuple containing the index of the
                original dataloader and the actual element (similar to Python's enumerate()).
        """

        self._data_loaders = data_loaders
        self._shuffle = shuffle
        self._return_dl_idx = return_dl_idx
        self._alternating_sampling = alternating_sampling

        if sampling_strategy is not None:
            self._sampling_strategy = sampling_strategy
        elif alternating_sampling:
            # sample from each data loader once before repeating
            self._sampling_strategy = CyclicSamplingStrategy(list(range(len(data_loaders))))
        elif shuffle:
            if sample_weights is None:
                # uniform random sampling
                self._sampling_strategy = RandomSamplingStrategy()
                # self._sample_weights = np.array([1 / len(data_loaders) for _ in range(len(data_loaders))])
            else:
                self._sampling_strategy = WeightedSamplingStrategy(sample_weights)
                # self._sample_weights = None if sample_weights is None else np.array(sample_weights) / sum(
                #     sample_weights)
                assert sample_weights is None or len(sample_weights) == len(data_loaders), \
                    f"Need to specify as many sample weights (got {len(sample_weights)}) " \
                    f"as dataloaders ({len(data_loaders)})"
        else:
            assert sample_weights is None, f"shuffle has to be set, if sample_weights are used"
            # self._sample_weights = None
            self._sampling_strategy = SequentialSamplingStrategy()

        self._stop_criterion = stop_criterion

    def save(self, data, **kwargs):
        raise Exception('CombinedDataManager cannot save')

    def __iter__(self) -> Iterator[_T]:
        return CombinedIterableDataLoader.Iterator([iter(data_manager) for data_manager in self._data_loaders],
                                                   self._sampling_strategy.create_sampler(len(self._data_loaders)),
                                                   self._stop_criterion,
                                                   self._return_dl_idx)

    class Iterator:

        def __init__(self,
                     iterators: List[Iterator[_T]],
                     data_loader_sampler: ChoiceSampler,
                     stop_criterion: CombinedIterableStopCriterion,
                     return_dl_idx: bool):
            self._iterators = iterators
            self._data_loader_sampler = data_loader_sampler
            self._stop_criterion = stop_criterion
            self._return_dl_idx = return_dl_idx
            self._last_chosen_iterator_idx = -1

            # self._identifiers = list(range(len(iterators)))

        def __next__(self) -> _T:
            if len(self._data_loader_sampler.get_remaining_choices()) == 0:
                raise StopIteration()

            iterator_idx = next(self._data_loader_sampler)

            # if self._alternating_sampling:
            #     idx = (self._last_chosen_iterator_idx + 1) % len(self._identifiers)
            #     iterator_idx = self._identifiers[idx]
            #     self._last_chosen_iterator_idx = idx
            # elif self._sample_weights is not None:
            #     sample_weights = self._sample_weights[self._identifiers]
            #     assert sum(sample_weights) > 0, f"sample_weights (initial: {self._sample_weights}) sum to 0"
            #     sample_weights /= sum(sample_weights)
            #     iterator_idx = np.random.choice(self._identifiers, p=sample_weights)
            # else:
            #     # Iterate through all iterators in order
            #     iterator_idx = self._identifiers[0]

            try:
                sample = next(self._iterators[iterator_idx])
                if self._return_dl_idx:
                    return iterator_idx, sample
                else:
                    return sample
            except StopIteration:
                # self._identifiers.remove(iterator_idx)
                # self._last_chosen_iterator_idx -= 1  # Ensure that alternating sampling will not jump over next iterator
                self._data_loader_sampler.choice_exhausted(iterator_idx)
                if self._stop_criterion.should_stop(iterator_idx, self._data_loader_sampler.get_remaining_choices()):
                    raise StopIteration()
                else:
                    return next(self)


class CombinedRandomAccessDataLoader(RandomAccessDataLoader[_T]):

    def __init__(self, data_loaders: List[RandomAccessDataLoader[_T]], shuffle=False):
        # TODO: sample_weights
        self._data_loaders = data_loaders
        self._shuffle = shuffle

        if shuffle:
            self._shuffled_indices = list(range(len(self)))
            np.random.shuffle(self._shuffled_indices)

    def __iter__(self) -> Iterator[_T]:
        return (self[idx] for idx in range(len(self)))

    def _get_single_item(self, idx: int) -> _T:
        assert -len(self) <= idx < len(
            self), f"Index {idx} is out of bounds for combined data loader of size {len(self)}"
        if self._shuffle:
            idx = self._shuffled_indices[idx]

        dl_idx, sample_idx = self._get_dl_idx_for_sample(idx)
        sample = self._data_loaders[dl_idx][sample_idx]

        return dl_idx, sample

    def __len__(self) -> int:
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