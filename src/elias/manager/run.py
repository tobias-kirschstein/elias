from pathlib import Path
from typing import Type


class RunManager:

    def __init__(self, location: str, run_name: str):

        run_location = f"{location}/{run_name}"
        assert Path(run_location).is_dir(), \
            f"Could not find directory '{location}'. Is the path correct?"

        self._location = run_location
        self._run_name = run_name

    @classmethod
    def from_location(cls: Type['RunManager'],
                      location: str,
                      run_name: str,
                      localize_via_run_name: bool = False) -> 'RunManager':
        """
        Creates a run manager for the specified location with default parameters.
        If the subclass constructor takes different arguments than the location, this needs to be overridden
        by subclasses to ensure that there is a instantiation method that takes exactly one argument.

        Parameters
        ----------
            location: path to the folder containing runs
            run_name: name of run that links to folder containing artifacts
            localize_via_run_name: whether only the run name is used to localize the folder

        Returns
        -------
            a run manager of the sub class at the specified location
        """

        try:
            if localize_via_run_name:
                run_manager = cls(run_name)
            else:
                run_manager = cls(location, run_name)
        except TypeError:
            raise NotImplementedError(f"Could not construct run manager {cls} with a single location parameter. "
                                      f"Please override from_location() to match the class __init__() method")
        return run_manager

    def get_run_name(self) -> str:
        return self._run_name
