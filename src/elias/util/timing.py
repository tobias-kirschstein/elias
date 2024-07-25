from collections import defaultdict
from statistics import mean, median
from time import time
from typing import Dict, Optional

from numpy import std
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
    def __init__(self,
                 max_iterations: Optional[int] = None,
                 warmup: int = 0,
                 print_at_last_iteration: bool = False,
                 disable: bool = False):
        """
        Utility to time repeated calculations in a loop and aggregate execution times.

        Parameters
        ----------
        max_iterations:
            If set, only the first n iterations will be recorded
        warmup:
            Optionally, skip the first n iterations before timings will be recorded (Often loops are slower in the beginning until caching kicks in)
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
        self.loop_start = None
        self.times = defaultdict(list)

        self._start_iteration = warmup
        self._end_iteration = max_iterations + warmup if max_iterations is not None else None
        self._current_iteration = 0
        self._print_at_last_iteration = print_at_last_iteration
        self._disable = disable

    def measure(self, name: str) -> Optional[float]:
        if self._disable:
            return None

        now = time()
        if self._start_iteration < self._current_iteration and (self._end_iteration is None or self._current_iteration < self._end_iteration):
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
        if not self._disable:
            self.start = time()

    def new_iteration(self):
        """
        Intended to be called at the beginning of a new iteration.
        """

        if not self._disable:
            self.start = time()

            # Note down how long the whole iteration took
            if (self.loop_start is not None
                    and self._start_iteration < self._current_iteration
                    and (self._end_iteration is None or self._current_iteration < self._end_iteration)):
                measured_loop_time = self.start - self.loop_start
                self.times["loop iteration"].append(measured_loop_time)

            self.loop_start = self.start
            self._current_iteration += 1

            if self._end_iteration is not None and self._current_iteration == self._end_iteration:
                self.print_summary()

    def get_iteration_time(self) -> float:
        """
        Returns
        -------
            the duration of the current loop iteration since new_iteration() in seconds
        """
        return time() - self.loop_start

    def summary(self) -> Dict[str, float]:
        if self._disable:
            return dict()

        summary = {name: mean(times) for name, times in self.times.items()}
        return summary

    def print_summary(self):
        if not self._disable:
            print("===========================")
            print("Timing Summary")
            print("===========================")
            summary = {name: [mean(times), median(times), std(times)] for name, times in self.times.items()}
            sorted_summary = sorted(summary.items(), key=lambda x: x[1], reverse=True)
            rows = []
            for name, (mean_time, median_time, std_time) in sorted_summary:
                row = [name, f"{median_time*1000:.3f}ms", f"{mean_time*1000:.3f}ms ± {std_time*1000:.3f}ms"]
                rows.append(row)

            print(tabulate(rows, headers=["section", "median", "mean+std"]))
