from pathlib import Path
from shutil import rmtree
from typing import Any, List, TypeVar, Generic, Dict, Tuple

import matplotlib.pyplot as plt

from elias.fs import ensure_file_ending, extract_file_numbering, ensure_directory_exists
from elias.generic import get_type_var_instantiation
from elias.io import save_pickled, load_pickled


class Analysis:

    def __init__(self, analysis_path: str, analysis_name: str):
        assert Path(f"{analysis_path}/{analysis_name}").is_dir(), \
            f"Could not find directory '{analysis_path}/{analysis_name}'. Is the name {analysis_name} correct?"

        self._analysis_name = analysis_name
        self._location = f"{analysis_path}/{self._analysis_name}"

    def save_pyplot_fig(self, fig_name: str):
        plt.savefig(f"{self._location}/{ensure_file_ending(fig_name, 'pdf')}")

    def save_object(self, obj: Any, name: str):
        save_pickled(obj, f"{self._location}/{name}")

    def save_objects(self, objects: Dict[str, Any]):
        for name, obj in objects.items():
            self.save_object(obj, name)

    def load_object(self, name: str) -> Any:
        return load_pickled(f"{self._location}/{name}")

    def load_objects(self, *names) -> Tuple[Any]:
        return tuple(self.load_object(name) for name in names)

    def ls(self) -> List[str]:
        return [p.name for p in Path(self._location).iterdir()]


_AnalysisType = TypeVar("_AnalysisType", bound=Analysis)


class AnalysisFolder(Generic[_AnalysisType]):

    def __init__(self, analysis_folder: str):
        assert Path(f"{analysis_folder}").is_dir(), \
            f"Could not find directory '{analysis_folder}'. Is the location correct?"

        self._location = analysis_folder
        self._analysis_cls = get_type_var_instantiation(self, _AnalysisType)

    def ls(self) -> List[str]:
        file_numberings = extract_file_numbering(self._location, r"(-?\d+)-.*")
        _, file_names = list(zip(*file_numberings))
        return file_names

    def cd(self, sub_folder: str) -> 'AnalysisFolder':
        return AnalysisFolder(f"{self._location}/{sub_folder}")

    def open(self, analysis_name: str) -> _AnalysisType:
        return self._analysis_cls(self._location, analysis_name)

    def new(self, analysis_name: str) -> _AnalysisType:
        file_numberings = extract_file_numbering(self._location, r"(-?\d+)-.*")
        if not file_numberings:
            analysis_name = f"1-{analysis_name}"
        else:
            file_numberings, _ = zip(*file_numberings)
            analysis_name = f"{max(file_numberings) + 1}-{analysis_name}"

        ensure_directory_exists(f"{self._location}/{analysis_name}")
        return self.open(analysis_name)

    def delete(self, analysis_name: str):
        rmtree(f"{self._location}/{analysis_name}")
