from pathlib import Path
from shutil import rmtree
from typing import List, Optional, Union

from elias.folder.folder import Folder


class RunFolder:
    _name_format: str
    _folder: Folder

    def __init__(self, location: str, name_format: str):
        self._folder = Folder(location)
        self._name_format = name_format

    def cd(self, sub_folder: str):
        # Inplace cd to avoid having to infer the correct subclass of the current object
        self._folder.cd(sub_folder, inplace=True)

    def list_runs(self) -> List[str]:
        return self._folder.list_file_numbering(self._name_format, return_only_file_names=True)

    def list_run_ids(self) -> List[int]:
        return self._folder.list_file_numbering(self._name_format, return_only_numbering=True)

    def generate_run_name(self, name: Optional[str] = None) -> str:
        return self._folder.generate_next_name(self._name_format, name=name)

    def delete_run(self, run_name_or_id: Union[str, int]):
        run_name = self.resolve_run_name(run_name_or_id)
        run_folder = f"{self._folder.get_location()}/{run_name}"
        assert Path(run_folder).exists(), f"Cannot delete run {run_name}. It does not exist"
        assert Path(run_folder).is_dir(), f"{run_folder} is not a folder"

        rmtree(run_folder)

    def get_run_name_by_id(self, run_id: int) -> Optional[str]:
        return self._folder.get_file_name_by_numbering(self._name_format, run_id)

    def resolve_run_name(self, run_name_or_id: Union[str, int]) -> Optional[str]:
        if isinstance(run_name_or_id, int):
            return self.get_run_name_by_id(run_name_or_id)
        else:
            return run_name_or_id
