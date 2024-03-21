import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Type, TypeVar, Generic, Optional, List, Union

from silberstral import reveal_type_var

from elias.manager.artifact import ArtifactManager, ArtifactType
from elias.config import Config
from elias.folder.folder import Folder

_ModelConfigType = TypeVar('_ModelConfigType', bound=Config)
_OptimizationConfigType = TypeVar('_OptimizationConfigType', bound=Config)
_DatasetConfigType = TypeVar('_DatasetConfigType', bound=Config)
_TrainSetupType = TypeVar('_TrainSetupType', bound=Config)
_EvaluationResultType = TypeVar('_EvaluationResultType', bound=Config)
_EvaluationConfigType = TypeVar('_EvaluationConfigType', bound=Config)
_ModelType = TypeVar('_ModelType')


class ModelManager(ABC,
                   Generic[
                       _ModelType, _ModelConfigType, _OptimizationConfigType, _DatasetConfigType, _TrainSetupType,
                       _EvaluationConfigType, _EvaluationResultType],
                   ArtifactManager):
    """
    Type Vars
    ---------

        _ModelType:
            The type of the model being handled by this model manager. Only there for type hinting, can be None
        _ModelConfigType:
            The class of the model configuration (stored in ``model_config.json``) which is assumed to be a dataclass
            subclassing :class:`elias.config.Config`. :meth:`save_model_config` and :meth:`load_model_config`
            take/retrieve the dataset configuration as a Python object of this class.
            Typical contents of a model_config:
                * Model Architecture
                * Model Type
                * Layers
                * Anything that affects learnable weights
                * Input/Output format
        _OptimizationConfigType:
            The class of the optimization configuration (stored in ``optimization_config.json``).
            Works analog to _ModelConfigType
            Typical contents of an optimization_config:
                * Loss
                * Regularization
                * Type of Optimizer (Adam, RMSProp, ...)
                * Learning Rate scheduling
        _DatasetConfigType:
            The class of the dataset configuration (stored in ``dataset_config.json``).
            Works analog to _ModelConfigType
            Typical contents of a dataset_config:
                * Dataset Type
                * Dataset Version
                * Partition
                * Image Augmentation
                * Any online Data Preprocessing
                * Anything that affects the format of batches fed to the model
        _TrainSetupType:
            The class of the train setup (stored in ``train_setup.json``).
            Works analog to _ModelConfigType
            Typical contents of a train_setup:
                * Seed
                * Batch size
                * train/valid split
                * Logging/Monitoring config
                * Multithreading
                * Gradient Accumulation
        _EvaluationConfigType:
            The class of the evaluation config.
            The file name depends on `evaluation_name_format` and `artifact_type`.
            E.g., if `evaluation_name_format="evaluation_epoch_$"` and JSON artifact the file name may be
            "evaluation_epoch_100_config.json".
            Can be None, if evaluation configs shall not be persisted
            Typical contents of an evaluation config:
                * Which metrics to evaluation
        _EvaluationResultType:
            The class of the evaluation result.
            The file name depends on `evaluation_name_format` and `artifact_type`.
            E.g., if `evaluation_name_format="evaluation_epoch_$"` and JSON artifact the file name may be
            "evaluation_epoch_100.json"
            Typical contents of an evaluation result:
                * Scores
    """

    _folder: Folder
    _run_name: str
    _checkpoint_name_format: Optional[str]
    _evaluation_name_format: Optional[str]
    _evaluation_config_name_format: Optional[str]
    _cls_model_config: Type[_ModelConfigType]
    _cls_optimization_config: Type[_OptimizationConfigType]
    _cls_dataset_config: Type[_DatasetConfigType]
    _cls_train_setup: Type[_TrainSetupType]
    _cls_evaluation_result: Type[_EvaluationResultType]
    _cls_evaluation_config: Type[_EvaluationConfigType]

    def __init__(self,
                 model_store_path: str,
                 run_name: str,
                 checkpoint_name_format: Optional[str] = None,
                 checkpoints_sub_folder: Optional[str] = None,
                 evaluation_name_format: Optional[str] = None,
                 artifact_type: ArtifactType = ArtifactType.JSON):
        """
        Creates a model manager that provides utilities for loading/storing checkpoints, configs and evaluations
        for a model run.

        Parameters
        ----------
            model_store_path:
                Root folder of runs
            run_name:
                Name of this specific run
            checkpoint_name_format:
                If storing/loading checkpoints via ID, e.g., epoch, is desired, a file name format has to be specified.
                It has to contain exactly one `$` indicating the checkpoint ID, e.g., `checkpoint-$.ckpt`
            checkpoints_sub_folder:
                In case the checkpoints are not stored in the main model folder, one can specify a sub folder within
                the main folder that will hold checkpoint files
            evaluation_name_format:
                If storing/loading evaluations via checkpoint ID is desired, a file name format has to be specified.
                It has to contain exactly one `$` indicating the checkpoint ID and EXCLUDING the file name suffix.
                This will be inferred automatically from the specified `artifact_type`.
                Example for `evaluation_name_format`: `evaluation_$`
            artifact_type:
                In which format, e.g., JSON or YAML, configs and evaluations should be stored
        """

        assert Path(
            f"{model_store_path}/{run_name}").is_dir(), f"Could not find directory '{model_store_path}/{run_name}'. Is the run name {run_name} correct?"
        super(ModelManager, self).__init__(f"{model_store_path}/{run_name}", artifact_type=artifact_type)

        self._folder = Folder(f"{model_store_path}/{run_name}")
        if checkpoints_sub_folder is None:
            self._checkpoints_folder = self._folder
        else:
            self._checkpoints_folder = Folder(f"{model_store_path}/{run_name}/{checkpoints_sub_folder}")

        self._run_name = run_name
        self._checkpoint_name_format = checkpoint_name_format

        if evaluation_name_format is not None:
            # Append correct artifact type suffix, e.g., .json or .yaml
            self._evaluation_name_format = f"{evaluation_name_format}.{self._artifact_type.get_file_ending()}"
            # Evaluation configs are hard-coded to have "_config" preceding the suffix
            self._evaluation_config_name_format = \
                f"{evaluation_name_format}_config.{self._artifact_type.get_file_ending()}"
        else:
            self._evaluation_name_format = None
            self._evaluation_config_name_format = None

        self._cls_model_config = reveal_type_var(self, _ModelConfigType)
        self._cls_optimization_config = reveal_type_var(self, _OptimizationConfigType)
        self._cls_dataset_config = reveal_type_var(self, _DatasetConfigType)
        self._cls_train_setup = reveal_type_var(self, _TrainSetupType)
        self._cls_evaluation_result = reveal_type_var(self, _EvaluationResultType)
        self._cls_evaluation_config = reveal_type_var(self, _EvaluationConfigType)

    @classmethod
    def from_location(cls: Type['ModelManager'],
                      model_store_path: str,
                      run_name: str,
                      localize_via_run_name: bool = False) -> 'ModelManager':
        """
        Creates a model manager for the specified location with default parameters.
        If the subclass constructor takes different arguments than the location, this needs to be overridden
        by subclasses to ensure that there is a instantiation method that takes exactly one argument.

        Parameters
        ----------
            model_store_path:
                path to the folder containing model runs
            run_name:
                name of the run
            localize_via_run_name:
                whether only the run name should be used to find model folder

        Returns
        -------
            a model manager for the specified run
        """

        try:
            if localize_via_run_name:
                new_model_manager = cls(run_name)
            else:
                new_model_manager = cls(model_store_path, run_name)
        except TypeError as e:
            raise NotImplementedError(f"Could not construct model manager {cls} with location and run name parameter. "
                                      f"Please override from_location() to match the class __init__() method"
                                      f"Other possible error: {e}")
        return new_model_manager

    # -------------------------------------------------------------------------
    # Building/Storing/Loading Models
    # -------------------------------------------------------------------------

    def get_checkpoint_path(self, checkpoint_name_or_id: Union[str, int]):
        assert self._checkpoint_name_format is not None, "Cannot get checkpoint path, no file name format specified"

        checkpoint_id = self._resolve_checkpoint_id(checkpoint_name_or_id)
        self._checkpoints_folder.substitute(self._checkpoint_name_format, checkpoint_id)
        checkpoint_file_name = self._checkpoints_folder.get_file_name_by_numbering(self._checkpoint_name_format,
                                                                                   checkpoint_id)

        checkpoint_path = f"{self._checkpoints_folder.get_location()}/{checkpoint_file_name}"
        return checkpoint_path

    def get_checkpoint_folder(self) -> str:
        return self._checkpoints_folder.get_location()

    def list_checkpoints(self) -> List[str]:
        assert self._checkpoint_name_format is not None, "Cannot list checkpoints, no file name format specified"

        checkpoint_ids_and_names = self._checkpoints_folder.list_file_numbering(self._checkpoint_name_format)
        if len(checkpoint_ids_and_names) == 0:
            return []

        checkpoint_ids, checkpoint_names = zip(*checkpoint_ids_and_names)
        last_neg_id = None
        for idx, checkpoint_id in enumerate(checkpoint_ids):
            if checkpoint_id < 0:
                last_neg_id = idx

        checkpoint_names = list(checkpoint_names)
        if last_neg_id is not None:
            checkpoint_names = checkpoint_names[last_neg_id + 1:] + checkpoint_names[:last_neg_id + 1]

        return checkpoint_names

    def list_checkpoint_ids(self) -> List[int]:
        assert self._checkpoint_name_format is not None, "Cannot list checkpoints, no file name format specified"

        checkpoint_ids = self._checkpoints_folder.list_file_numbering(self._checkpoint_name_format,
                                                                      return_only_numbering=True)
        last_neg_id = None
        for idx, checkpoint_id in enumerate(checkpoint_ids):
            if checkpoint_id < 0:
                last_neg_id = idx

        if last_neg_id is not None:
            checkpoint_ids = checkpoint_ids[last_neg_id + 1:] + checkpoint_ids[:last_neg_id + 1]

        return checkpoint_ids

    def store_checkpoint(self, model: _ModelType, checkpoint_name_or_id: Union[str, int], **kwargs):
        if isinstance(checkpoint_name_or_id, int):
            assert self._checkpoint_name_format is not None, \
                f"Cannot store checkpoint with id {checkpoint_name_or_id} since no checkpoint name format was specified"
            checkpoint_file_name = self._checkpoints_folder.substitute(self._checkpoint_name_format,
                                                                       checkpoint_name_or_id)
        else:
            checkpoint_file_name = checkpoint_name_or_id

        self._store_checkpoint(model, checkpoint_file_name, **kwargs)

    def load_checkpoint(self, checkpoint_name_or_id: Union[str, int], **kwargs) -> _ModelType:
        checkpoint_id = self._resolve_checkpoint_id(checkpoint_name_or_id)
        checkpoint_file_name = self._checkpoints_folder.get_file_name_by_numbering(self._checkpoint_name_format,
                                                                                   checkpoint_id)

        return self._load_checkpoint(checkpoint_file_name, **kwargs)

    def delete_checkpoint(self, checkpoint_name_or_id: Union[str, int]):
        checkpoint_id = self._resolve_checkpoint_id(checkpoint_name_or_id)
        checkpoint_file_name = self._checkpoints_folder.get_file_name_by_numbering(self._checkpoint_name_format,
                                                                                   checkpoint_id)

        os.remove(f"{self._checkpoints_folder}/{checkpoint_file_name}")

    def build_model(self,
                    model_config: Optional[_ModelConfigType] = None,
                    optimization_config: Optional[_OptimizationConfigType] = None,
                    load_optimization_config: bool = False,
                    **kwargs) -> _ModelType:
        model_config = self.load_model_config() if model_config is None else model_config
        optimization_config = self.load_optimization_config() \
            if optimization_config is None and load_optimization_config \
            else optimization_config

        return self._build_model(model_config, optimization_config, **kwargs)

    @abstractmethod
    def _build_model(self,
                     model_config: _ModelConfigType,
                     optimization_config: Optional[_OptimizationConfigType] = None,
                     **kwargs) -> _ModelType:
        """
        Should be overwritten by subclass to construct an actual model object given the model and optimization configs

        Parameters
        ----------
            model_config:
                the configuration of the model to construct
            optimization_config:
                specifies what kind of optimization the model should be prepared for

        Returns
        -------
            A fully operable model with the given configs. No trained parameters have been applied yet
        """

        pass

    @abstractmethod
    def _store_checkpoint(self, model: _ModelType, checkpoint_file_name: str, **kwargs):
        """
        Should be overwritten by subclass to specify how exactly the model will be persisted

        Parameters
        ----------
            model:
                The model to persist
            checkpoint_file_name:
                The full name for the checkpoint excluding the path to the folder
        """

        pass

    @abstractmethod
    def _load_checkpoint(self, checkpoint_file_name: Union[str, int], **kwargs) -> _ModelType:
        """
        Should be overwritten by subclass to specify how exactly a persisted checkpoint should be loaded.
        Typically, this method will make use of `_build_model()` and then apply the loaded trained weights from the
        checkpoint file.

        Parameters
        ----------
            checkpoint_file_name:
                name of the file containing learned weights

        Returns
        -------
            A fully operable model with the learned weights from the specified checkpoint file
        """

        pass

    # -------------------------------------------------------------------------
    # Storing/Loading Configs
    # -------------------------------------------------------------------------

    def store_model_config(self, model_config: _ModelConfigType):
        self._save_artifact(model_config.to_json(), "model_config")

    def load_model_config(self) -> _ModelConfigType:
        return self._load_config(self._cls_model_config, "model_config")

    def store_optimization_config(self, optimization_config: _OptimizationConfigType):
        self._save_artifact(optimization_config.to_json(), "optimization_config")

    def load_optimization_config(self) -> _OptimizationConfigType:
        return self._load_config(self._cls_optimization_config, "optimization_config")

    def store_dataset_config(self, dataset_config: _DatasetConfigType):
        self._save_artifact(dataset_config.to_json(), "dataset_config")

    def load_dataset_config(self) -> _DatasetConfigType:
        return self._load_config(self._cls_dataset_config, "dataset_config")

    def store_train_setup(self, train_setup: _TrainSetupType):
        self._save_artifact(train_setup.to_json(), "train_setup")

    def load_train_setup(self) -> _TrainSetupType:
        return self._load_config(self._cls_train_setup, "train_setup")

    # -------------------------------------------------------------------------
    # Evaluations
    # -------------------------------------------------------------------------

    def list_evaluations(self) -> List[str]:
        assert self._evaluation_name_format is not None, "Cannot list evaluations, no file name format specified"
        evaluation_files = self._folder.list_file_numbering(self._evaluation_name_format, return_only_file_names=True)

        return evaluation_files

    def list_evaluated_checkpoint_ids(self) -> List[int]:
        assert self._evaluation_name_format is not None, "Cannot list evaluations, no file name format specified"
        checkpoint_ids = self._folder.list_file_numbering(self._evaluation_name_format, return_only_numbering=True)

        return checkpoint_ids

    def store_evaluation_result(self, evaluation_result: _EvaluationResultType, checkpoint_name_or_id: Union[str, int]):
        if isinstance(checkpoint_name_or_id, str) and self._evaluation_name_format is None:
            # Assume that given checkpoint name should be file name for evaluation
            self._save_artifact(evaluation_result.to_json(), checkpoint_name_or_id)
        else:
            assert self._evaluation_name_format is not None, "Cannot store evaluation, no file name format specified"
            checkpoint_id = self._resolve_checkpoint_id(checkpoint_name_or_id)
            evaluation_file_name = self._folder.substitute(self._evaluation_name_format, checkpoint_id)
            self._save_artifact(evaluation_result.to_json(), evaluation_file_name)

    def load_evaluation_result(self, checkpoint_name_or_id: Union[str, int]) -> _EvaluationResultType:
        if isinstance(checkpoint_name_or_id, str) and self._evaluation_name_format is None:
            # Assume that given checkpoint name should be file name for evaluation
            return self._load_config(self._cls_evaluation_result, checkpoint_name_or_id)
        else:
            assert self._evaluation_name_format is not None, "Cannot load evaluation, no file name format specified"

            checkpoint_id = self._resolve_checkpoint_id(checkpoint_name_or_id)
            evaluation_file_name = self._folder.substitute(self._evaluation_name_format, checkpoint_id)
            return self._load_config(self._cls_evaluation_result, evaluation_file_name)

    def store_evaluation_config(self, evaluation_config: _EvaluationConfigType, checkpoint_name_or_id: Union[str, int]):
        assert self._evaluation_config_name_format is not None, \
            "Cannot store evaluation config, no file name format specified"
        checkpoint_id = self._resolve_checkpoint_id(checkpoint_name_or_id)
        evaluation_file_name = self._folder.substitute(self._evaluation_config_name_format, checkpoint_id)
        self._save_artifact(evaluation_config.to_json(), evaluation_file_name)

    def load_evaluation_config(self, checkpoint_name_or_id: Union[str, int]):
        assert self._evaluation_config_name_format is not None, \
            "Cannot load evaluation config, no file name format specified"

        checkpoint_id = self._resolve_checkpoint_id(checkpoint_name_or_id)
        evaluation_file_name = self._folder.substitute(self._evaluation_config_name_format, checkpoint_id)
        return self._load_config(self._cls_evaluation_config, evaluation_file_name)

    # -------------------------------------------------------------------------
    # Getters
    # -------------------------------------------------------------------------

    def get_run_name(self) -> str:
        return self._run_name

    def get_location(self) -> str:
        return self._folder.get_location()

    def get_model_store_path(self) -> str:
        return self._folder.get_location()

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    def _load_config(self, artifact_cls: Type[Config], artifact_name: str) -> Config:
        if isinstance(None, artifact_cls):
            raise NotImplementedError(f"Cannot load `{artifact_name}` as its corresponding type is not defined")
        return artifact_cls.from_json(self._load_artifact(artifact_name))

    def _resolve_checkpoint_id(self, checkpoint_name_or_id: Union[str, int]) -> int:
        # Special handling for allowing 'latest' and 'last' as checkpoint names
        if checkpoint_name_or_id in {'latest', 'last'}:
            checkpoint_name_or_id = -1

        if isinstance(checkpoint_name_or_id, int):
            assert self._checkpoint_name_format is not None, \
                "Cannot resolve checkpoint by id, no file name format specified"
            if checkpoint_name_or_id < 0:
                # Find latest checkpoint
                assert checkpoint_name_or_id == -1, \
                    f"Only -1 is allowed as negative checkpoint id, got `{checkpoint_name_or_id}`"

                checkpoint_ids, checkpoint_names = zip(
                    *self._checkpoints_folder.list_file_numbering(self._checkpoint_name_format))
                if -1 in checkpoint_ids:
                    checkpoint_id = -1
                else:
                    # No checkpoint with id -1 present => Take checkpoint with largest ID instead
                    assert len(checkpoint_ids) > 0, \
                        f"Cannot find latest checkpoint, no checkpoints found in {self._checkpoints_folder.get_location()}"
                    checkpoint_id = checkpoint_ids[-1]
            else:
                # Find corresponding name for checkpoint id
                checkpoint_id = checkpoint_name_or_id

            return checkpoint_id
        else:
            checkpoint_id = self._checkpoints_folder.get_numbering_by_file_name(self._checkpoint_name_format,
                                                                                checkpoint_name_or_id)
            return checkpoint_id
