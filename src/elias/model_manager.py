from abc import ABC, abstractmethod
from pathlib import Path
from typing import Type, TypeVar, Generic, Optional, List

from elias.config import Config
from elias.fs import list_file_numbering, generate_run_name, create_directories
from elias.generic import get_type_var_instantiation
from elias.io import save_json, load_json

ModelConfigType = TypeVar('ModelConfigType', bound=Config)
TrainConfigType = TypeVar('TrainConfigType', bound=Config)
DatasetConfigType = TypeVar('DatasetConfigType', bound=Config)
TrainSetupType = TypeVar('TrainSetupType', bound=Config)
ModelType = TypeVar('ModelType')


class ModelManager(ABC, Generic[ModelType, ModelConfigType, TrainConfigType, DatasetConfigType, TrainSetupType]):

    def __init__(self,
                 model_store_path: str,
                 run_name: str):
        assert Path(
            f"{model_store_path}/{run_name}").is_dir(), f"Could not find directory '{model_store_path}/{run_name}'. Is the run name {run_name} correct?"
        self._model_store_path = f"{model_store_path}/{run_name}"
        self._run_name = run_name
        self._cls_model_config: Type[ModelConfigType] = get_type_var_instantiation(self, ModelConfigType)
        self._cls_train_config: Type[TrainConfigType] = get_type_var_instantiation(self, TrainConfigType)
        self._cls_dataset_config: Type[DatasetConfigType] = get_type_var_instantiation(self, DatasetConfigType)
        self._cls_train_setup: Type[TrainSetupType] = get_type_var_instantiation(self, TrainSetupType)

    @abstractmethod
    def store_checkpoint(self, model: Type[ModelType], checkpoint_name: str):
        pass

    @abstractmethod
    def _load_checkpoint(self, checkpoint_name: str, map_location=None) -> ModelType:
        pass

    # TODO: how do we want to have the Checkpoint Manager?
    def load_checkpoint(self, checkpoint_name: str, map_location=None) -> ModelType:
        # TODO: add assertion for map_location
        if checkpoint_name in {'latest', 'last'}:
            checkpoint_name = -1
        if isinstance(checkpoint_name, int) and checkpoint_name < 0:
            checkpoints = self.list_checkpoints()
            checkpoint_name = checkpoints[checkpoint_name]

        return self._load_checkpoint(checkpoint_name, map_location)

    @abstractmethod
    def list_checkpoints(self) -> List[str]:
        pass

    @abstractmethod
    def _build_model(self, model_config: ModelConfigType, train_config: Optional[TrainConfigType] = None) -> ModelType:
        pass

    def build_model(self,
                    model_config: Optional[ModelConfigType] = None,
                    train_config: Optional[TrainConfigType] = None,
                    load_train_config: bool = False) -> ModelType:
        model_config = self.load_model_config() if model_config is None else model_config
        train_config = self.load_train_config() if train_config is None and load_train_config else train_config
        return self._build_model(model_config, train_config)

    def store_model_config(self, model_config: ModelConfigType):
        self._store_artifact(model_config, "model_config")

    def load_model_config(self) -> ModelConfigType:
        return self._load_artifact(self._cls_model_config, "model_config")

    def store_train_config(self, train_config: TrainConfigType):
        self._store_artifact(train_config, "train_config")

    def load_train_config(self) -> TrainConfigType:
        return self._load_artifact(self._cls_train_config, "train_config")

    def store_dataset_config(self, dataset_config: DatasetConfigType):
        self._store_artifact(dataset_config, "dataset_config")

    def load_dataset_config(self) -> DatasetConfigType:
        return self._load_artifact(self._cls_dataset_config, "dataset_config")

    def store_train_setup(self, train_setup: TrainSetupType):
        self._store_artifact(train_setup, "train_setup")

    def load_train_setup(self) -> TrainSetupType:
        return self._load_artifact(self._cls_train_setup, "train_setup")

    def get_run_name(self) -> str:
        return self._run_name

    def get_model_store_path(self) -> str:
        return self._model_store_path

    def _store_artifact(self, artifact: Config, artifact_name: str):
        # TODO: should it be allows to store artifacts even if the corresponding type is not defined ?
        save_json(artifact.to_json(), f"{self._model_store_path}/{artifact_name}")

    def _load_artifact(self, artifact_cls: Type[Config], artifact_name: str) -> Config:
        if isinstance(None, artifact_cls):
            raise NotImplementedError(f"Cannot load `{artifact_name}` as its corresponding type is not defined")
        return artifact_cls.from_json(load_json(f"{self._model_store_path}/{artifact_name}"))


ModelManagerType = TypeVar("ModelManagerType", bound=ModelManager)


class RunManager(Generic[ModelManagerType]):

    def __init__(self,
                 runs_dir: str,
                 prefix: str):
        self._runs_dir = runs_dir
        self._prefix = prefix

    def list_runs(self) -> List[str]:
        run_ids = list_file_numbering(self._runs_dir, f"{self._prefix}-")
        return [f"{self._prefix}-{run_id}" for run_id in run_ids]

    def generate_run_name(self) -> str:
        return generate_run_name(self._runs_dir, self._prefix, match_arbitrary_suffixes=True)

    def new_run(self) -> ModelManagerType:
        run_name = self.generate_run_name()
        create_directories(f"{self._runs_dir}/{run_name}/")
        return self.get_model_manager(run_name)

    @abstractmethod
    def get_model_manager(self, run_name) -> ModelManagerType:
        pass
