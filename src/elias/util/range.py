import re
from abc import abstractmethod
from typing import Union, List

_RANGE_EXTRACTOR = re.compile(r"(-?\d+)-(-?\d+)")
_STEP_RANGE_EXTRACTOR = re.compile(r"(\d+)n(\+(\d+))?")


class ImplementsContains:

    @abstractmethod
    def __contains__(self, idx: int) -> bool:
        pass


class StepRange(ImplementsContains):
    step_size: int
    offset: int = 0

    def __init__(self, step_size: int, offset: int = 0):
        self.step_size = step_size
        self.offset = offset

    def __contains__(self, idx: int) -> bool:
        return (idx - self.offset) % self.step_size == 0


class IndexRange:
    """
    Represents an interval that is closed at both ends: [a, b].
    Useful for parsing user-defined ranges from the command line, e.g., 10-25.
    Negative numbers can be passed as well, in which case it is necessary to call `resolve()` with the actual number
    of items that the indices within the IndexRange refer to.
    E.g., 5--1 will resolve to 5-19 for a sequence of 20 items, or 5-99 for a sequence of 100 items.

    Provides parsing intervals from textual range specifications:
     - 1-5 -> [1, 5]
     - 1 -> [1, 1]
     - 7--1 -> resolve(20) -> [7, 19]
    """

    start_idx: int
    end_idx: int
    resolved: bool

    def __init__(self, start_idx: int, end_idx: int):
        assert start_idx <= 0 <= end_idx \
               or end_idx <= 0 <= start_idx \
               or start_idx <= end_idx <= 0 \
               or 0 <= start_idx <= end_idx, \
            "start_idx has to be smaller than end_idx"

        self.start_idx = start_idx
        self.end_idx = end_idx
        self.resolved = start_idx >= 0 and end_idx >= 0

    @staticmethod
    def from_description(index_range_description: str) -> Union['IndexRange', StepRange]:
        try:
            # Just a single number n. This is ok, will lead to an interval [n, n] that only accepts that number
            number = int(index_range_description)
            return IndexRange(number, number)
        except ValueError:
            pass

        step_range = _STEP_RANGE_EXTRACTOR.match(index_range_description)
        if step_range:
            step_size = int(step_range.group(1))  # The '5' of 5n+3
            offset = step_range.group(3)  # The '3' of 5n+3
            if offset is None:
                offset = 0
            else:
                offset = int(offset)
            assert 0 <= offset < step_size, f"offset should be between 0 and step_size {step_size}. Got {offset}"
            return StepRange(step_size, offset)

        frame_range = _RANGE_EXTRACTOR.search(index_range_description)
        # Ensure to convert negative indices to actual list indices
        start_idx = int(frame_range.group(1))
        end_idx = int(frame_range.group(2))

        return IndexRange(start_idx, end_idx)

    def resolve(self, n_samples: int) -> 'IndexRange':
        assert n_samples > abs(self.end_idx) and n_samples > abs(self.start_idx), \
            f"n_samples of {n_samples} is too small for index range [{self.start_idx}: {self.end_idx}]"

        self.start_idx = self.start_idx % n_samples
        self.end_idx = self.end_idx % n_samples
        self.resolved = True

        return self

    def __contains__(self, idx: int) -> bool:
        assert self.resolved, "Can only use 'in' on resolved index ranges"
        return self.start_idx <= idx <= self.end_idx

    def __len__(self) -> int:
        assert self.resolved, "Can only compute length on resolved index ranges"

        return self.end_idx - self.start_idx + 1  # + 1 because right end of interval is inclusive

    def __iter__(self):
        assert self.resolved, "Cannot iterate an unresolved index range"

        return iter(range(self.start_idx, self.end_idx + 1))


class IndexRangeBundle:

    def __init__(self, index_ranges: List[IndexRange]):
        self._index_ranges = index_ranges

    @staticmethod
    def from_description(range_specifier: str) -> 'IndexRangeBundle':
        range_specifier = range_specifier.replace("\\", "")  # Get rid of escape characters \\
        index_ranges = [IndexRange.from_description(description) for description in
                        range_specifier.split(',')]
        return IndexRangeBundle(index_ranges)

    def resolve(self, n_samples: int):
        for index_range in self._index_ranges:
            index_range.resolve(n_samples)

    def __contains__(self, idx: int) -> bool:
        for index_range in self._index_ranges:
            if idx in index_range:
                return True

        return False
