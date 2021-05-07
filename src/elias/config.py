from __future__ import annotations

from dataclasses import dataclass, asdict

from dacite import from_dict


@dataclass
class Config:

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, json_config: dict) -> Config:
        return from_dict(cls, json_config)
