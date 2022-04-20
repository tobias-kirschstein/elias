from abc import ABC, abstractmethod
from typing import Iterable, Iterator, Union, List, Set, TypeVar

_T = TypeVar('_T')


class IterableDataLoader(ABC):
    @abstractmethod
    def __iter__(self):
        pass


class RandomAccessDataLoader(Iterable[_T]):

    def __iter__(self) -> Iterator[_T]:
        return (self[idx] for idx in range(len(self)))

    def __getitem__(self, idx: Union[int, slice]) -> Union[_T, List[_T]]:
        if isinstance(idx, slice):
            return self.get_slice(idx)
        elif isinstance(idx, int):
            return self._get_single_item(idx)
        else:
            try:
                idx_int = int(idx)
                return self._get_single_item(idx_int)
            except TypeError:
                raise ValueError(f"Unsupported index type passed to __getitem__: {type(idx)}")

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def _get_single_item(self, idx: int) -> _T:
        pass

    def get_slice(self, index_slice: slice) -> List[_T]:
        """
        Retrieves the specified slice from the dataloader. As opposed to `view()`, the items are not lazyly loaded but
        directly retrieved.

        Parameters
        ----------
            index_slice: The indices (in form of a slice) to retrieve

        Returns
        -------
            a list containing the requested items
        """

        return [self[idx] for idx in range(*index_slice.indices(len(self)))]

    def view(self,
             indices: Union[slice, List[int], Set[int]],
             exclude: bool = False) -> 'RandomAccessDataLoaderView[_T]':
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

        # TODO: find a way to inherit self completely and basically only change the __get_item__() method
        return RandomAccessDataLoaderView(self, indices, exclude=exclude)


class RandomAccessDataLoaderView(RandomAccessDataLoader[_T]):
    """
    Provides simple means of changing the iteration over a dataloader without copying the underlying data.
    """

    def __init__(self,
                 dataloader: RandomAccessDataLoader[_T],
                 indices: Union[slice, List[int], Set[int]],
                 exclude: bool = False):
        """
        Parameters
        ----------
            dataloader: the underlying dataloader which is viewed at
            indices: the indices of the respective elements that will be used by the dataloader view. When the view
                is iterated over, the order of passed indices will be used.
                If a set is passed, the order of indices is undefined.
        """

        self._dataloader = dataloader
        n_samples = len(self._dataloader)
        if isinstance(indices, slice):
            ranged_indices = range(n_samples)[indices]
            # TODO: DataLoader can only be iterated over once
            if exclude:
                ranged_indices = set(ranged_indices)
                self._indices = (idx for idx in range(n_samples) if idx not in ranged_indices)
            else:
                self._indices = ranged_indices
        elif isinstance(indices, list) or isinstance(indices, set):
            assert not indices or max(indices) < n_samples, \
                f"Cannot create view with index {max(indices)} for data loader with length {n_samples}"
            if exclude:
                indices = indices if isinstance(indices, set) else set(indices)
                self._indices = [idx for idx in range(n_samples) if idx not in indices]
            else:
                self._indices = list(indices)
        else:
            raise ValueError(f"view indices must be slice or list not {type(indices)}")

    def __iter__(self) -> Iterator[_T]:
        return (self._dataloader[idx] for idx in self._indices)

    def _get_single_item(self, idx: int) -> _T:
        return self._dataloader[self._indices[idx]]

    def __len__(self) -> _T:
        return len(self._indices)
