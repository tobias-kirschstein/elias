from abc import ABC, abstractmethod
from typing import Iterator, List

import numpy as np


class ChoiceSampler(ABC, Iterator[int]):

    def __init__(self, n_choices: int):
        self._choices = list(range(n_choices))

    def choice_exhausted(self, choice_idx: int):
        self._choices.remove(choice_idx)

    def get_remaining_choices(self) -> List[int]:
        return self._choices


class SamplingStrategy(ABC):

    @abstractmethod
    def create_sampler(self, n_choices: int) -> ChoiceSampler:
        pass


class RandomSamplingStrategy(SamplingStrategy):
    class RandomChoiceSampler(ChoiceSampler):

        def __next__(self) -> int:
            return np.random.choice(self._choices)

    def create_sampler(self, n_choices: int) -> ChoiceSampler:
        return self.RandomChoiceSampler(n_choices)


class WeightedSamplingStrategy(SamplingStrategy):
    class WeightedSampler(ChoiceSampler):

        def __init__(self, n_choices: int, weights: List[float]):
            assert n_choices == len(weights), \
                f"WeightedSamplingStrategy got {len(weights)} weights, but {n_choices} choices"
            super(WeightedSamplingStrategy.WeightedSampler, self).__init__(n_choices)
            weights = np.array(weights)

            # Completely ignore choices that were given 0 probability
            zero_prob_choices = weights == 0
            weights = weights[~zero_prob_choices]
            self._choices = [choice
                             for choice, zero_prob_choice in zip(self._choices, zero_prob_choices)
                             if not zero_prob_choice]

            self._original_weights = weights / sum(weights)
            self._weights = self._original_weights

        def __next__(self) -> int:
            return np.random.choice(self._choices, p=self._weights)

        def choice_exhausted(self, choice_idx: int):
            super(WeightedSamplingStrategy.WeightedSampler, self).choice_exhausted(choice_idx)

            if len(self.get_remaining_choices()) == 0:
                return

            remaining_weights = self._original_weights[self._choices]
            remaining_weights /= sum(remaining_weights)
            assert sum(remaining_weights) > 0, f"remaining weights (initial: {self._original_weights}) sum to 0"

            self._weights = remaining_weights

    def __init__(self, weights: List[float]):
        self._weights = weights

    def create_sampler(self, n_choices: int) -> ChoiceSampler:
        return self.WeightedSampler(n_choices, self._weights)


class CyclicSamplingStrategy(SamplingStrategy):
    class CyclicSampler(ChoiceSampler):

        def __init__(self, n_choices: int, sample_sequence: List[int]):
            assert min(sample_sequence) >= 0 and max(sample_sequence) < n_choices, \
                f"sample sequence ({sample_sequence}) contains choices that are out of range for {n_choices} choices"
            super().__init__(n_choices)
            self._sample_sequence = list(sample_sequence)
            self._current_idx = 0

        def __next__(self) -> int:
            choice = self._sample_sequence[self._current_idx]
            self._current_idx = (self._current_idx + 1) % len(self._sample_sequence)
            return choice

        def choice_exhausted(self, choice_idx: int):
            super(CyclicSamplingStrategy.CyclicSampler, self).choice_exhausted(choice_idx)

            if len(self.get_remaining_choices()) == 0:
                return

            # 1, 2, 2, 1, 3
            #       x
            # remove 2
            # 1, 1, 3
            #    x

            # Remove all occurrences of the choice from the sample sequence
            filtered_sample_sequence = list()
            for idx, choice in enumerate(self._sample_sequence):
                if idx < self._current_idx:
                    # Ensure that pointer is moved in such a way that sampling proceeds as if choice_idx didn't exist
                    self._current_idx -= 1
                if choice != choice_idx:
                    filtered_sample_sequence.append(choice)

            # Can happen that all choices after current_idx were deleted. Then pointer needs to get back to beginning
            self._current_idx = self._current_idx % len(filtered_sample_sequence)

            self._sample_sequence = filtered_sample_sequence

    def __init__(self, sample_sequence: List[int]):
        self._sample_sequence = sample_sequence

    def create_sampler(self, n_choices: int) -> ChoiceSampler:
        return self.CyclicSampler(n_choices, self._sample_sequence)


class SequentialSamplingStrategy(SamplingStrategy):
    class SequentialSampler(ChoiceSampler):
        def __next__(self) -> int:
            return self._choices[0]

    def create_sampler(self, n_choices: int) -> ChoiceSampler:
        return self.SequentialSampler(n_choices)