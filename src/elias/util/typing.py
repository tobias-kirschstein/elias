from importlib import import_module
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


def class_to_module_path(cls: Type) -> str:
    module = cls.__module__
    if module == 'builtins':
        module_path = cls.__qualname__  # avoid outputs like 'builtins.str'
    else:
        module_path = module + '.' + cls.__qualname__

    return module_path


def module_path_to_class(dotted_path: str) -> Type:
    """
    Import a dotted module path and return the attribute/class designated by the
    last name in the path. Raise ImportError if the import failed.
    """

    parts = dotted_path.rsplit('.', 1)
    if len(parts) == 2:
        module_path = parts[0]
        class_name = parts[1]
    else:
        module_path = 'builtins'
        class_name = parts[0]

    module = import_module(module_path)
    return getattr(module, class_name)
