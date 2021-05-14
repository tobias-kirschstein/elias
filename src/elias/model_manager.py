from abc import ABC, abstractmethod
from typing import Type, TypeVar, Generic, Optional, List

from elias.config import Config
from elias.fs import list_file_numbering, generate_run_name
from elias.io import save_json, load_json

ModelConfigType = TypeVar('ModelConfigType', bound=Config)
TrainConfigType = TypeVar('TrainConfigType', bound=Config)
DatasetConfigType = TypeVar('DatasetConfigType', bound=Config)
ModelType = TypeVar('ModelType')


class ModelManager(ABC, Generic[ModelType, ModelConfigType, TrainConfigType, DatasetConfigType]):

    def __init__(self,
                 model_store_path: str,
                 run_name: str,
                 cls_model_config: Type[ModelConfigType],
                 cls_train_config: Type[TrainConfigType],
                 cls_dataset_config: Type[DatasetConfigType]):
        self._model_store_path = f"{model_store_path}/{run_name}"
        self._run_name = run_name
        self._cls_model_config = cls_model_config
        self._cls_train_config = cls_train_config
        self._cls_dataset_config = cls_dataset_config

    @abstractmethod
    def store_checkpoint(self, model: Type[ModelType], checkpoint_name: str):
        pass

    @abstractmethod
    def _load_checkpoint(self, checkpoint_name: str, map_location=None) -> ModelType:
        pass

    # TODO: how do we want to have the Checkpoint Manager?
    def load_checkpoint(self, checkpoint_name: str, map_location=None) -> ModelType:
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
        save_json(model_config.to_json(), f"{self._model_store_path}/model_config")

    def load_model_config(self) -> ModelConfigType:
        return self._cls_model_config.from_json(load_json(f"{self._model_store_path}/model_config"))

    def store_train_config(self, train_config: TrainConfigType):
        save_json(train_config.to_json(), f"{self._model_store_path}/train_config")

    def load_train_config(self) -> TrainConfigType:
        return self._cls_train_config.from_json(load_json(f"{self._model_store_path}/train_config"))

    def store_dataset_config(self, dataset_config: DatasetConfigType):
        save_json(dataset_config.to_json(), f"{self._model_store_path}/dataset_config")

    def load_dataset_config(self) -> DatasetConfigType:
        return self._cls_dataset_config.from_json(load_json(f"{self._model_store_path}/dataset_config"))

    def get_run_name(self) -> str:
        return self._run_name

    def get_model_store_path(self) -> str:
        return self._model_store_path


class RunManager:

    def __init__(self,
                 runs_dir: str,
                 prefix: str,
                 cls_model_manager: Type[ModelManager],
                 cls_model_config: Type[ModelConfigType],
                 cls_train_config: Type[TrainConfigType],
                 cls_dataset_config: Type[DatasetConfigType]):
        self._runs_dir = runs_dir
        self._prefix = prefix
        self._cls_model_manager = cls_model_manager
        self._cls_model_config = cls_model_config
        self._cls_train_config = cls_train_config
        self._cls_dataset_config = cls_dataset_config

    def list_runs(self) -> List[str]:
        run_ids = list_file_numbering(self._runs_dir, f"{self._prefix}-")
        return [f"{self._prefix}-{run_id}" for run_id in run_ids]

    def generate_run_name(self) -> str:
        return generate_run_name(self._runs_dir, self._prefix, match_arbitrary_suffixes=True)

    def _create_model_manager(self,
                              runs_dir: str,
                              run_name: str,
                              cls_model_config: Type[ModelConfigType],
                              cls_train_config: Type[TrainConfigType],
                              cls_dataset_config: Type[DatasetConfigType]) -> ModelManager:
        return self._cls_model_manager(runs_dir, run_name, cls_model_config, cls_train_config, cls_dataset_config)

    def new_run(self) -> ModelManager:
        return self._create_model_manager(self._runs_dir,
                                          self.generate_run_name(),
                                          self._cls_model_config,
                                          self._cls_train_config,
                                          self._cls_dataset_config)

    def get_model_manager(self, run_name) -> ModelManager:
        return self._create_model_manager(self._runs_dir,
                                          run_name,
                                          self._cls_model_config,
                                          self._cls_train_config,
                                          self._cls_dataset_config)
