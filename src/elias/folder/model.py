from shutil import rmtree
from typing import Type, Union, Optional

from silberstral.silberstral import create_linked_type_var

from elias.folder.run import RunFolder, _RunManagerType
from elias.manager.model import ModelManager

_ModelManagerType = create_linked_type_var(_RunManagerType, bound=ModelManager)


class ModelFolder(RunFolder[_ModelManagerType]):
    _cls_run_manager: Type[_ModelManagerType]
    _prefix: str

    def __init__(self, models_folder: str, prefix: str, localize_via_run_name: bool = False):
        self._prefix = prefix
        super(ModelFolder, self).__init__(models_folder, f"{prefix}-$[-*]", localize_via_run_name=localize_via_run_name)

    def get_prefix(self) -> str:
        return self._prefix

    def open_run(self, run_name_or_id: Union[str, int]) -> _ModelManagerType:
        run_name = self.resolve_run_name(run_name_or_id)
        return self._cls_run_manager.from_location(self._folder.get_location(),
                                                   run_name,
                                                   localize_via_run_name=self._localize_via_run_name)

    def new_run(self, name: Optional[str] = None) -> _ModelManagerType:
        new_run_name = self.generate_run_name(name)
        return self.open_run(new_run_name)

    def delete_run(self, run_name_or_id: Union[str, int]):
        run_name = self.resolve_run_name(run_name_or_id)
        rmtree(f"{self._folder.get_location()}/{run_name}")
