from pathlib import Path
from shutil import rmtree
from typing import List, Optional, Union, Generic, TypeVar, Type

from silberstral import reveal_type_var

from elias.folder.folder import Folder
from elias.manager.run import RunManager
from elias.util import ensure_directory_exists

_RunManagerType = TypeVar('_RunManagerType', bound=RunManager)


class RunFolder(Generic[_RunManagerType]):
    """
    Manages several runs stored in a folder. A run can be the result of any arbitrary script.
    Typically the artifacts of several runs of the same script would be stored in the same run folder.
    Each run is identified by an incrementing run id.
    Runs in the same run folder share a name format that can contain the following wildcards:
        - $: for the run id
        - *: for extra name information for a run (optional)

    A typical name format might look like this: RUN-$[-*]
    """

    _name_format: str
    _folder: Folder

    def __init__(self, location: str, name_format: str, localize_via_run_name: bool = False):
        ensure_directory_exists(location)
        self._folder = Folder(location)
        self._name_format = name_format
        self._localize_via_run_name = localize_via_run_name

        self._cls_run_manager: Type[_RunManagerType] = reveal_type_var(self, _RunManagerType)

    def get_location(self) -> str:
        return self._folder.get_location()

    def cd(self, sub_folder: str):
        # Inplace cd to avoid having to infer the correct subclass of the current object
        self._folder.cd(sub_folder, inplace=True)

    def list_runs(self) -> List[str]:
        return self._folder.list_file_numbering(self._name_format, return_only_file_names=True)

    def list_run_ids(self) -> List[int]:
        return self._folder.list_file_numbering(self._name_format, return_only_numbering=True)

    def generate_run_name(self, name: Optional[str] = None) -> str:
        return self._folder.generate_next_name(self._name_format, name=name)

    def open_run(self, run_name_or_id: Union[str, int]) -> _RunManagerType:
        run_name = self.resolve_run_name(run_name_or_id)
        return self._cls_run_manager.from_location(self._folder.get_location(),
                                                   run_name,
                                                   localize_via_run_name=self._localize_via_run_name)

    def new_run(self, name: Optional[str] = None) -> _RunManagerType:
        new_run_name = self.generate_run_name(name)
        return self.open_run(new_run_name)

    def delete_run(self, run_name_or_id: Union[str, int]):
        run_name = self.resolve_run_name(run_name_or_id)
        run_folder = f"{self._folder.get_location()}/{run_name}"
        assert Path(run_folder).exists(), f"Cannot delete run {run_name}. It does not exist"
        assert Path(run_folder).is_dir(), f"{run_folder} is not a folder"

        rmtree(run_folder)

    def get_run_name_by_id(self, run_id: int) -> Optional[str]:
        return self._folder.get_file_name_by_numbering(self._name_format, run_id)

    def get_run_id_by_name(self, run_name: str) -> Optional[int]:
        return self._folder.get_numbering_by_file_name(self._name_format, run_name)

    def substitute(self, run_id: int, name: Optional[str] = None) -> str:
        """
        Returns the run name in the correct format for this run folder.
        E.g., format is "RUN-$[-*]"
        substitute(2, name="test") -> RUN-2-test

        Parameters
        ----------
            run_id: the run id that should be substituted into the name format
            name: Optionally, a descriptive name of the run that should be substituted into the name format

        Returns
        -------
            A run name following the name format of this run folder
        """

        return self._folder.substitute(self._name_format, run_id, name=name)

    def resolve_run_name(self, run_name_or_id: Union[str, int]) -> Optional[str]:
        """
        Find complete run name given a partial run name or run ID.

        :param run_name_or_id:
            for example, RUN-2 or 2
        :return:
            The complete run name that was found in the run folder, e.g., RUN-2-higher_LR
        """

        if isinstance(run_name_or_id, int):
            return self.get_run_name_by_id(run_name_or_id)
        elif self._folder.file_exists(run_name_or_id):
            # Run name was given in its entirety
            return run_name_or_id
        else:
            # Maybe run name was given without optional parts, e.g., NET-123 instead of NET-123-more-dropout
            # Try to extract run id and then find full file name from that
            file_numbering = self._folder.get_numbering_by_file_name(self._name_format, run_name_or_id)
            if file_numbering is None:
                # Given run_name does not match name format
                return None

            file_name = self._folder.get_file_name_by_numbering(self._name_format, file_numbering)
            return file_name
