from pathlib import Path
from typing import Any, List, Dict, Tuple, Type

import matplotlib.pyplot as plt

from elias.util.fs import ensure_file_ending
from elias.util.io import save_pickled, load_pickled


class Analysis:
    _location: str

    def __init__(self, location: str, analysis_name: str):
        analysis_location = f"{location}/{analysis_name}"
        assert Path(analysis_location).is_dir(), \
            f"Could not find directory '{location}'. Is the path correct?"

        self._location = analysis_location

    @classmethod
    def from_location(cls: Type['Analysis'],
                      location: str,
                      analysis_name: str,
                      localize_via_analysis_name: bool = False) -> 'Analysis':
        """
        Creates an analysis for the specified location with default parameters.
        If the subclass constructor takes different arguments than the location, this needs to be overridden
        by subclasses to ensure that there is a instantiation method that takes exactly one argument.

        Parameters
        ----------
            location: path to the folder containing analyses
            analysis_name: name of analysis linking to folder containing analysis artifacts
            localize_via_analysis_name: whether only the analysis name should be used to find the folder

        Returns
        -------
            an analysis of the sub class at the specified location
        """

        try:
            if localize_via_analysis_name:
                new_analysis = cls(analysis_name)
            else:
                new_analysis = cls(location, analysis_name)
        except TypeError:
            raise NotImplementedError(f"Could not construct analysis {cls} with a single location parameter. "
                                      f"Please override from_location() to match the class __init__() method")
        return new_analysis

    def save_pyplot_fig(self, fig_name: str, file_ending: str = 'pdf'):
        plt.savefig(f"{self._location}/{ensure_file_ending(fig_name, file_ending)}")

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


