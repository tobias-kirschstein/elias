from typing import Type, List


def ensure_type(obj, cls: Type):
    """
    Ensures that `obj` is of type `cls` by calling `cls`'s constructor in case it isn't.
    """
    if not isinstance(obj, cls):
        return cls(obj)
    else:
        return obj


def ensure_list(obj) -> List:
    if not isinstance(obj, list):
        return [obj]
    else:
        return obj
