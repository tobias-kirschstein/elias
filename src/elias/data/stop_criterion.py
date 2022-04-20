from abc import abstractmethod
from typing import List


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
