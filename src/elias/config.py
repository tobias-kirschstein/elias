from __future__ import annotations

import inspect
from abc import ABC
from dataclasses import dataclass, asdict, fields
from enum import Enum, EnumMeta, auto
from typing import List, Tuple, Any, Type

import dacite
from dacite import from_dict


# =========================================================================
# Better Enum handling for persistable config objects
# =========================================================================

class NamedEnumMeta(EnumMeta):
    """
    Enum Meta to enable instantiating enums with their name as well.
    """

    def __call__(cls: NamedEnum, value, *args, **kw):
        if isinstance(value, str):
            value = cls.from_name(value).value
        return super().__call__(value, *args, **kw)


class NamedEnum(Enum, metaclass=NamedEnumMeta):
    """
    Simple enum improvement that allows instantiating enums not only by their internal value, but also by their name.
    This comes handy when parsing enums from string values that come from the command line or a configuration file.

    Examples
    --------
        >>> class Color(NamedEnum):
        >>>     RED = 0
        >>>     YELLOW = 1
        >>> assert(Color(0) == Color('RED'))
        >>> assert(Color('red') == Color('rEd'))
    """

    @classmethod
    def from_name(cls, name: str):
        for k, v in cls.__members__.items():
            if k == name.upper():
                return v
        raise ValueError(f"Could not find `{name}` in enum {cls}")


class StringEnum(str, NamedEnum):
    """
    Simple enum extension that uses the string names of the members as values instead of integer values.
    One can simply employ the :meth:`auto()` method in the definition of the enum.
    This comes in handy when serializing an enum as the stored values will be descriptive (i.e., human readable)

    Examples
    --------
        >>> class Color(StringEnum):
        >>>     RED = auto()
        >>>     YELLOW = auto()
        >>> assert(isinstance(Color.RED.name), str)
        >>> assert(Color.RED.name == 'RED')
    """

    def _generate_next_value_(name, start, count, last_values):
        return name


# =========================================================================
# Actual Config class
# =========================================================================

@dataclass
class Config(ABC):

    @staticmethod
    def _serialize_enums(items: List[Tuple[str, Any]]):
        d = dict()
        for key, value in items:
            if isinstance(value, Enum):
                d[key] = value.value
            else:
                d[key] = value
        return d

    def to_json(self) -> dict:
        """
        Converts this configuration dataclass into an ordinary Python dictionary that can be easy persisted as JSON.
        Special attention is given to enum members of the dataclass. As enums cannot be serialized per default, these
        are represented by their intrinsic value instead. When deserializing the stored JSON with :meth:`from_json`
        the enum values can be parsed into proper enums again thanks to the type annotation in the underlying dataclass.

        Returns
        -------
            a Python dictionary representing this dataclass
        """
        return asdict(self, dict_factory=Config._serialize_enums)

    def __post_init__(self):
        """
        Overrides the post initialization hook of Python's dataclasses.
        This default implementation scans for fields in the dataclass that have an enum type hint.
        Any of these attributes that are not proper enums of their type will then be automatically instantiated.
        This comes handy when initializing a Config from command line values as these are often only strings hinting
        at the enum member.

        """
        casts = self.__class__._define_casts()
        for field in fields(self):
            field_value = getattr(self, field.name)
            if field.type in casts and not isinstance(field_value, field.type):
                setattr(self, field.name, field.type(field_value))

    @classmethod
    def _define_casts(cls) -> List[Type]:
        """
        Lists all the types of dataclass attributes that are enums and thus should be casted to their proper types
        if they not yet belong to it.

        Returns
        -------
            a list of types for which explicit conversion is initiated whenever this Config dataclass is instantiated

        """
        casts = []
        for field in fields(cls):
            field_type = field.type if inspect.isclass(field.type) else type(field.type)
            if issubclass(field_type, Enum):
                casts.append(field_type)
        return casts

    @classmethod
    def from_json(cls, json_config: dict) -> Config:
        """
        Constructs this Config dataclass from the given Python dictionary which typically will be a parsed JSON.
        As enums are not serialized in JSONs, special attention is put to such attributes.
        Any enum value that were stored as strings in the JSON file will be explicitly converted to their respective
        enum type.

        Parameters
        ----------
        json_config: dict
            the dictionary representing the JSON configuration. if the dictionary contains keys that don't match the
            dataclass an exception will be thrown. If you want to ignore excess items, see :meth:`from_dict`

        Returns
        -------
            This dataclass with all the values from :attr:`json_config` filled in. Enum attributes are explicitly
            converted.

        """
        return from_dict(cls, json_config, config=dacite.Config(cast=cls._define_casts()))

    @classmethod
    def from_dict(cls, values: dict):
        """
        Attempts to construct this Config by filling in the appropriate values from the specified :attr:`values`.
        Any excess items in the passed dictionary are ignored allowing to instantiate the Config dataclass with
        a bigger dictionary without having to hand-select the matching keys.
        Does not initiate explicit type conversion for enum attributes.

        Parameters
        ----------
            values: dict
                the dictionary holding the config items. Only matching keys will be used for instantiating the config
                object

        Returns
        -------
            The filled config object
        """

        return cls(**{
            k: v for k, v in values.items()
            if k in inspect.signature(cls).parameters
        })


class DotDict(dict):
    """
    Simple extension of Python's dict to support dot access.
    """

    def __init__(self, *args, **kwargs):
        super(DotDict, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.items():
                if isinstance(v, dict):
                    self[k] = DotDict(**v)
                else:
                    self[k] = v

    def __getattr__(self, attr):
        return self[attr]

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(DotDict, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(DotDict, self).__delitem__(key)
        del self.__dict__[key]
