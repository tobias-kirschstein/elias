from enum import Enum, auto
from pathlib import Path
from typing import Callable

from elias.util.io import save_json, save_yaml, load_json, load_yaml


class ArtifactType(Enum):
    JSON = auto()
    YAML = auto()

    def get_saver(self) -> Callable[[dict, str], None]:
        if self == ArtifactType.JSON:
            return save_json
        elif self == ArtifactType.YAML:
            return save_yaml
        else:
            raise ValueError(f"Unknown Artifact type: {self}")

    def get_loader(self) -> Callable[[str], dict]:
        if self == ArtifactType.JSON:
            return load_json
        elif self == ArtifactType.YAML:
            return load_yaml
        else:
            raise ValueError(f"Unknown Artifact type: {self}")

    def get_file_ending(self) -> str:
        if self == ArtifactType.JSON:
            return 'json'
        elif self == ArtifactType.YAML:
            return 'yaml'
        else:
            raise ValueError(f"Unknown Artifact type: {self}")


class ArtifactManager:

    def __init__(self, location: str, artifact_type: ArtifactType = ArtifactType.JSON):
        assert Path(location).is_dir(), f"Specified location '{location}' is not a directory"

        self._location = location
        self._artifact_type = artifact_type

    def _save_artifact(self, artifact: dict, name: str):
        self._artifact_type.get_saver()(artifact, f"{self._location}/{name}")

    def _load_artifact(self, name: str) -> dict:
        return self._artifact_type.get_loader()(f"{self._location}/{name}")
