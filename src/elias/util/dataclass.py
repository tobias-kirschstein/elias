from dataclasses import fields, dataclass
from typing import Type


def better_asdict(config) -> dict:
    items = dict()
    for f in fields(config):
        if f.init:
            field_value = getattr(config, f.name)
            items[f.name] = field_value

    return items


def sub_dataclass(cls: Type, config, **kwargs):
    """
    Solves the problem that dataclasses can have fields with init=False which can not be initialized directly in the constructor.

    Parameters
    ----------
    cls: Type of the dataclass subclass
    config: parent dataclass instantiation
    kwargs: constructor arguments for the dataclass subclass

    Returns
    -------


    """
    items = better_asdict(config)
    sub_object = cls(**items, **kwargs)
    for f in fields(config):
        if not f.init:
            field_value = getattr(config, f.name)
            setattr(sub_object, f.name, field_value)

    return sub_object
