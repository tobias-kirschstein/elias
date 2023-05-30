from pathlib import Path
from typing import Type, Generic, TypeVar

from silberstral import reveal_type_var

from elias.config import Config
from elias.manager import ArtifactManager
from elias.manager.artifact import ArtifactType

_ConfigType = TypeVar('_ConfigType', bound=Config)


# TODO: Rethink the use of run_name in the managers
#   For optional wildcards, we cannot resolve the run_name here because we don't have the name_format (only folder has)
#   Maybe instantiation only via folders? But how to handle additional parameters in the __init__ for the manager?
class RunManager(Generic[_ConfigType], ArtifactManager):

    def __init__(self, location: str, run_name: str, artifact_type=ArtifactType.JSON):
        run_location = f"{location}/{run_name}"
        assert Path(run_location).is_dir(), \
            f"Could not find directory '{run_location}'. Is the path correct?"
        super(RunManager, self).__init__(run_location, artifact_type=artifact_type)

        self._location = run_location
        self._run_name = run_name
        self._config_cls: Config = reveal_type_var(self, _ConfigType)

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

    def load_config(self) -> _ConfigType:
        json_config = self._load_artifact("config")
        return self._config_cls.from_json(json_config)

    def save_config(self, config: _ConfigType):
        self._save_artifact(config.to_json(), "config")

    def get_location(self) -> str:
        return self._location

    def get_run_name(self) -> str:
        return self._run_name
