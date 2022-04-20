from shutil import rmtree
from typing import TypeVar, Generic, Type, Union

from silberstral import reveal_type_var

from elias.manager.model import ModelManager
from elias.folder.run import RunFolder

_ModelManagerType = TypeVar("_ModelManagerType", bound=ModelManager)


class ModelFolder(Generic[_ModelManagerType], RunFolder):
    _model_manager_cls: Type[_ModelManagerType]

    def __init__(self, models_folder: str, prefix: str):
        super(ModelFolder, self).__init__(models_folder, f"{prefix}-$")
        self._model_manager_cls = reveal_type_var(self, _ModelManagerType)

    def open_run(self, run_name_or_id: Union[str, int]) -> _ModelManagerType:
        run_name = self.resolve_run_name(run_name_or_id)
        return self._model_manager_cls.from_location(self._folder.get_location(), run_name)

    def new_run(self) -> _ModelManagerType:
        new_run_name = self.generate_run_name()
        return self.open_run(new_run_name)

    def delete_run(self, run_name_or_id: Union[str, int]):
        run_name = self.resolve_run_name(run_name_or_id)
        rmtree(f"{self._folder.get_location()}/{run_name}")