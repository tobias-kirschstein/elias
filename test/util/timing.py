from time import sleep
from unittest import TestCase

from elias.util.timing import LoopTimer


class TimingTest(TestCase):
    def test_loop_timer(self):
        n_iterations = 10
        sleep_time = 0.0001
        loop_timer = LoopTimer(max_iterations=n_iterations)

        for i in range(100):
            loop_timer.new_iteration()
            sleep(sleep_time * i)
            loop_timer.measure("sleep")
            j = i + 1
            loop_timer.measure("add")

        summary = loop_timer.summary()
        self.assertGreater(summary['sleep'], ((sleep_time * 0) + (sleep_time * n_iterations)) / 2)