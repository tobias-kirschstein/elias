from dataclasses import dataclass
from typing import List, Union, Optional
from unittest import TestCase

from testfixtures import TempDirectory

from elias.config import Config
from elias.manager.artifact import ArtifactType
from elias.manager.model import ModelManager
from elias.util.io import save_pickled, load_pickled
from elias.folder import ModelFolder


@dataclass
class Model:
    weights: List[float]


@dataclass
class ModelConfig(Config):
    n_layers: int


@dataclass
class OptimizationConfig(Config):
    loss: str


@dataclass
class TrainSetup(Config):
    seed: int


@dataclass
class DatasetConfig(Config):
    dataset_version: str


@dataclass
class EvaluationConfig(Config):
    evaluate_mse: bool


@dataclass
class EvaluationResult(Config):
    mse: float


class TestModelManager(ModelManager[Model, ModelConfig, OptimizationConfig, DatasetConfig, TrainSetup,
                                    EvaluationConfig, EvaluationResult]):

    def __init__(self, model_store_location: str, run_name: str):
        super(TestModelManager, self).__init__(model_store_location,
                                               run_name,
                                               checkpoint_name_format="checkpoint-$.p",
                                               evaluation_name_format="evaluation_$")

    def _build_model(self,
                     model_config: ModelConfig,
                     optimization_config: Optional[OptimizationConfig] = None,
                     **kwargs) -> Model:
        return Model([0 for _ in range(model_config.n_layers)])

    def _store_checkpoint(self,
                          model: Model,
                          checkpoint_file_name: str, **kwargs):
        save_pickled(model.weights, f"{self.get_model_store_path()}/{checkpoint_file_name}")

    def _load_checkpoint(self,
                         checkpoint_file_name: Union[str, int],
                         **kwargs) -> Model:
        weights: List[float] = load_pickled(f"{self.get_model_store_path()}/{checkpoint_file_name}")
        model = self.build_model()
        for idx, weight in enumerate(weights):
            model.weights[idx] = weights[idx]

        return model


class SimpleModelManager(ModelManager[None, ModelConfig, OptimizationConfig, None, None, None, None]):

    def _build_model(self, model_config: Model,
                     optimization_config: Optional[OptimizationConfig] = None, **kwargs) -> Model:
        return Model([])

    def _store_checkpoint(self, model: Model, checkpoint_file_name: str, **kwargs):
        pass

    def _load_checkpoint(self, checkpoint_file_name: Union[str, int], **kwargs) -> Model:
        pass


class TestModelFolder(ModelFolder[TestModelManager]):

    def __init__(self, models_folder: str):
        super(TestModelFolder, self).__init__(models_folder, "RUN")


class ModelManagerTest(TestCase):

    def setUp(self) -> None:
        d = TempDirectory()
        d.makedir("RUN-9")
        d.makedir("RUN-10")

        self._model_config = ModelConfig(3)
        self._optimization_config = OptimizationConfig("L1")
        self._train_setup = TrainSetup(42)
        self._dataset_config = DatasetConfig("v1.0")
        self._evaluation_config = EvaluationConfig(True)
        self._evaluation_result = EvaluationResult(0.99)

        self._checkpoint_50 = [1.1, 2.2, 3.3]
        self._checkpoint_100 = [50, 25, 0]
        self._checkpoint_neg1 = [-1, -1, -1]

        json_saver = ArtifactType.JSON.get_saver()
        json_saver(self._model_config.to_json(), f"{d.path}/RUN-9/model_config.json")
        json_saver(self._optimization_config.to_json(), f"{d.path}/RUN-9/optimization_config.json")
        json_saver(self._train_setup.to_json(), f"{d.path}/RUN-9/train_setup.json")
        json_saver(self._dataset_config.to_json(), f"{d.path}/RUN-9/dataset_config.json")
        json_saver(self._evaluation_config.to_json(), f"{d.path}/RUN-9/evaluation_50_config.json")
        json_saver(self._evaluation_result.to_json(), f"{d.path}/RUN-9/evaluation_-1.json")

        save_pickled(self._checkpoint_50, f"{d.path}/RUN-9/checkpoint-50.p")
        save_pickled(self._checkpoint_100, f"{d.path}/RUN-9/checkpoint-100.p")
        save_pickled(self._checkpoint_neg1, f"{d.path}/RUN-9/checkpoint--1.p")

        self._directory = d
        self._model_folder = TestModelFolder(d.path)
        self._model_manager = self._model_folder.open_run('RUN-9')

    def tearDown(self) -> None:
        self._directory.cleanup()

    def test_list_runs(self):
        self.assertEqual(self._model_folder.list_run_ids(), [9, 10])
        self.assertEqual(self._model_folder.list_runs(), ['RUN-9', 'RUN-10'])

    def test_list_checkpoints(self):
        self.assertEqual(self._model_manager.list_checkpoint_ids(), [50, 100, -1])
        self.assertEqual(self._model_manager.list_checkpoints(),
                         ['checkpoint-50.p', 'checkpoint-100.p', 'checkpoint--1.p'])

    def test_list_evaluations(self):
        self.assertEqual(self._model_manager.list_evaluated_checkpoint_ids(), [-1])
        self.assertEqual(self._model_manager.list_evaluations(), ['evaluation_-1.json'])

    def test_load_configs(self):
        model_config = self._model_manager.load_model_config()
        optimization_config = self._model_manager.load_optimization_config()
        train_setup = self._model_manager.load_train_setup()
        dataset_config = self._model_manager.load_dataset_config()
        evaluation_config = self._model_manager.load_evaluation_config(50)
        evaluation_result = self._model_manager.load_evaluation_result('latest')

        self.assertEqual(model_config, self._model_config)
        self.assertEqual(optimization_config, self._optimization_config)
        self.assertEqual(train_setup, self._train_setup)
        self.assertEqual(dataset_config, self._dataset_config)
        self.assertEqual(evaluation_config, self._evaluation_config)
        self.assertEqual(evaluation_result, self._evaluation_result)

    def test_load_checkpoints(self):
        dummy_model = self._model_manager.build_model(ModelConfig(10))
        self.assertEqual(len(dummy_model.weights), 10)

        self.assertEqual(self._model_manager.load_checkpoint(50).weights, self._checkpoint_50)
        self.assertEqual(self._model_manager.load_checkpoint('checkpoint-100.p').weights, self._checkpoint_100)
        self.assertEqual(self._model_manager.load_checkpoint('last').weights, self._checkpoint_neg1)

    def test_store_checkpoint(self):
        dummy_model = self._model_manager.build_model()
        dummy_model.weights[0] += 2.5
        dummy_model.weights[1] -= 2.5

        self._model_manager.store_checkpoint(dummy_model, 25)
        loaded_model = self._model_manager.load_checkpoint(25)
        self.assertEqual(dummy_model, loaded_model)

    def test_store_evaluation(self):
        evaluation_config = EvaluationConfig(evaluate_mse=False)
        evaluation_result = EvaluationResult(mse=-1)

        self._model_manager.store_evaluation_config(evaluation_config, 'last')
        self._model_manager.store_evaluation_result(evaluation_result, 'latest')

        self.assertEqual(self._model_manager.load_evaluation_config('latest'), evaluation_config)
        self.assertEqual(self._model_manager.load_evaluation_result(-1), evaluation_result)

    def test_simple_model_manager(self):
        simple_model_manager = SimpleModelManager(self._directory.path, 'RUN-9')

        simple_model_manager.store_model_config(self._model_config)
        simple_model_manager.store_optimization_config(self._optimization_config)

        with self.assertRaises(NotImplementedError):
            simple_model_manager.load_dataset_config()

        with self.assertRaises(NotImplementedError):
            simple_model_manager.load_train_setup()

        with self.assertRaises(AssertionError):
            simple_model_manager.load_evaluation_result(-1)

        with self.assertRaises(AssertionError):
            simple_model_manager.load_evaluation_config(-1)
