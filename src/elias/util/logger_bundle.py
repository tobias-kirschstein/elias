from argparse import Namespace
from statistics import mean
from typing import List, Union, Dict, Any, Optional

import numpy as np

try:
    import torch
except ImportError:
    print("torch not installed")


class LoggerBundle:

    def __init__(self, loggers: List['Logger'] = None, accumulate: Optional[int] = None, initial_step: int = -1):
        """
        Logs metrics and hyperparams to all specified loggers

        Parameters
        ----------
            loggers: specific logger instances to log to
            accumulate: If set, metrics will accumulate for the specified number of steps before being sent to the actual logger instance
        """

        self._loggers = loggers or []
        self._accumulate_steps = accumulate
        self._cached_metrics = dict()  # metric => List[float]
        self._ready_metrics = dict()  # metric => float, metrics that are ready to be uploaded upon next step increase
        self._latest_step_per_metric = dict()  # metric => int
        self._current_step = initial_step

    def set_step(self, step: int):
        assert step >= self._current_step
        self._current_step = step

    def next_step(self):
        self.update_step(self._current_step + 1)

    def update_step(self, step: Optional[int] = None) -> int:
        step = self._current_step if step is None else step
        previous_step = self._current_step
        self.set_step(step)

        if step > previous_step > -1:
            if self._accumulate_steps is None or int(step / self._accumulate_steps) > int(previous_step / self._accumulate_steps):
                self.flush_metrics()

        return step

    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]):
        """
        Logs hyperparams to all specified loggers.
        """
        for logger in self._loggers:
            logger.log_hyperparams(params)

    def flush_metrics(self):
        if not self._loggers:
            return

        # Aggregate all cached metrics and mark them to be submitted
        metrics_to_log = dict()
        for metric_name, metric_values in self._cached_metrics.items():
            if metric_values:
                metrics_to_log[metric_name] = mean(metric_values)
            self._cached_metrics[metric_name] = []

        if metrics_to_log:
            from pytorch_lightning.loggers import Logger, WandbLogger, TensorBoardLogger
            for logger in self._loggers:
                if isinstance(logger, WandbLogger):
                    logger.experiment.log(metrics_to_log, step=self._current_step, commit=False)  # commit=False is necessary, otherwise internal step is automatically increased by wandb and one gets warnings about not able to log to previous step
                else:
                    logger.log_metrics(metrics_to_log, step=self._current_step)

    def log_metrics(self, metrics: Dict[str, Union[float, torch.Tensor]], step: Optional[int] = None, flush: bool = False):
        """
        Logs metrics to all specified loggers. This method logs metrics as soon as it received them.

        Parameters
        ----------
            metrics: metrics dict to log
            step: which step to log to. If none, the logger assumes that the step has not increased since the last time a step was logged
        """

        if not self._loggers:
            return

        cleaned_metrics = dict()
        for metric_name, metric_value, in metrics.items():
            if isinstance(metric_value, torch.Tensor):
                cleaned_metrics[metric_name] = metric_value.mean().item()
            elif isinstance(metric_value, list) and len(metric_value) > 0:
                cleaned_metrics[metric_name] = mean(metric_value)
            elif isinstance(metric_value, float) or isinstance(metric_value, int):
                cleaned_metrics[metric_name] = metric_value
            else:
                print(f"Dropping metric {metric_name} as its value {metric_value} is either empty or has an unsupported format")

        self.update_step(step)

        for metric, value in cleaned_metrics.items():
            # last_metric_update_step = self._latest_step_per_metric[metric] if metric in self._latest_step_per_metric else -1

            if metric not in self._cached_metrics:
                self._cached_metrics[metric] = []

            # if int(step / self._accumulate_steps) > int(last_metric_update_step / self._accumulate_steps):
            #     # Potentially aggregate cached metrics and mark them to be submitted
            #     if len(self._cached_metrics[metric]) > 0:
            #         assert metric not in self._ready_metrics
            #         self._ready_metrics[metric] = mean(self._cached_metrics[metric])
            #     self._cached_metrics[metric] = []

            # Cache current metric value
            self._cached_metrics[metric].append(value)
            # self._latest_step_per_metric[metric] = step

        if flush:
            self.flush_metrics()

    def log_image(self, key: str, images: List[Any], step: Optional[int] = None):
        if not self._loggers:
            return

        from pytorch_lightning.loggers import Logger, WandbLogger, TensorBoardLogger
        from wandb import Image
        step = self.update_step(step)
        for logger in self._loggers:
            if isinstance(logger, WandbLogger):
                # wandb logging
                # logger.experiment.log({key: [Image(img) for img in images]}, step=step, commit=True)
                logger.experiment.log({key: [Image(img) for img in images]}, step=step)
            elif isinstance(logger, TensorBoardLogger):
                # tensorboard logging
                if isinstance(images[0], np.ndarray):
                    images = np.stack(images)
                    input_format = "NHWC"
                elif isinstance(images[0], torch.Tensor):
                    images = torch.stack(images)
                    input_format = "NCHW"
                else:
                    raise ValueError(f"Unexpected type {type(images[0])}")
                logger.experiment.add_images(key, images, global_step=step, dataformats=input_format)


def has_passed(period: int, current_step: int, batch_size: int = 1) -> bool:
    """
    Computes whether the current step has passed the periodic marker.
    This method is especially helpful when current step is increased by larger batch sizes where trivial checks such
    as current_step % period == 0 would not yield the desired behavior as they might miss triggering the event.

    Parameters
    ----------
        period: frequency of a given event (in steps), i.e., logging eval metrics or storing a checkpoint
        current_step: current iteration to check whether it is time to trigger the event
        batch_size: How much the current step is increased in each iteration

    Returns
    -------
        Whether it is time for the periodic event according to the current step
    """

    previous_step = current_step - batch_size
    return int(current_step / period) > int(previous_step / period)
