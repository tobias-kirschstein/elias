from time import time


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
        self.time_measurer = Timing.TimeMeasurer()
        return self.time_measurer

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.time_measurer.measure()
