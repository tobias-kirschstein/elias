from collections import defaultdict
from statistics import mean
from time import time
from typing import Dict, Optional

from tabulate import tabulate


class TimeMeasurer:

    def __init__(self):
        self.start = time()
        self.times = []

    def measure(self) -> float:
        now = time()
        measured_time = now - self.start
        self.times.append(measured_time)
        self.start = now
        return measured_time

    def print_measure(self, text: str):
        passed_time = self.measure()
        print(f"{text}: {passed_time:.2f}ms")

    def __len__(self) -> int:
        return len(self.times)

    def __getitem__(self, i) -> float:
        return self.times[i]

    def __str__(self) -> str:
        if len(self.times) == 1:
            return f"{self.times[0]:.2f}"
        return f"{self.times}"


class Timing:
    """
    Usage:

    with Timing() as t:
        ...
    print(f"took {t[0]:0.2} seconds")

    Within the with-statement we can take several measurements by calling t.measure().
    Every measurement will be available after the closure in the list t.
    """

    def __enter__(self) -> TimeMeasurer:
        self.time_measurer = TimeMeasurer()
        return self.time_measurer

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.time_measurer.measure()


class LoopTimer:
    def __init__(self, max_iterations: Optional[int] = None, print_at_last_iteration: bool = False):
        """
        Utility to time repeated calculations in a loop and aggregate execution times.

        Parameters
        ----------
        max_iterations:
            If set, only the first n iterations will be recorded
        print_at_last_iteration:
            If set, a timing summary will be printed automatically once new_iteration() is called for the first time after max_iterations has been reached

        Usage
        -----
        >>> timer = LoopTimer
        >>> for i in range(100):
        >>>     timer.new_iteration()
        >>>     cmd1()
        >>>     timer.measure("cmd1")
        >>>     cmd2()
        >>>     timer.measure("cmd2")
        >>>
        >>> timer.print_summary()
        """

        self.start = time()
        self.times = defaultdict(list)

        self._max_iterations = max_iterations
        self._current_iteration = 0
        self._print_at_last_iteration = print_at_last_iteration

    def measure(self, name: str) -> Optional[float]:
        now = time()
        if self._max_iterations is None or self._current_iteration < self._max_iterations:
            measured_time = now - self.start
            self.times[name].append(measured_time)
        else:
            measured_time = None
        self.start = now
        return measured_time

    def start(self):
        """
        Set start time for next measurement to now.
        """

        self.start = time()

    def new_iteration(self):
        """
        Intended to be called at the beginning of a new iteration.
        """

        self.start = time()
        self._current_iteration += 1

        if self._max_iterations is not None and self._current_iteration == self._max_iterations:
            self.print_summary()

    def summary(self) -> Dict[str, float]:
        summary = {name: mean(times) for name, times in self.times.items()}
        return summary

    def print_summary(self):
        print("===========================")
        print("Timing Summary")
        print("===========================")
        summary = self.summary()
        rows = []
        for name, avg_time in summary.items():
            row = [name, f"{avg_time*1000:.3f}ms"]
            rows.append(row)

        print(tabulate(rows))
