from typing import get_args, Type, Union, get_origin, TypeVar, Generic, Iterable, Set

from dacite.types import is_generic_collection, extract_generic, is_union


def get_type_var_instantiation(obj_or_cls: Union[object, Type], type_var: TypeVar):
    """
    Obtains the corresponding class instantiation of the given `type_var` for the passed `obj_or_cls`.

    In Python, one can inherit from a Generic[T] class where T is a type var, to hint templated classes.
    However, it is not possible to use the actual class that is passed for T when an instance is created.
    Instead, one would have to explicitly pass the class of the type var to the constructor.
    To avoid this dualism of class declarations for type hints and actual types which one can instantiate, this
    utility method allows retrieving of the actual type instance for a type var.

    Examples
    --------
        >>> T = TypeVar("T")
        >>> class TemplatedClass(Generic[T]):
        >>>     pass
        >>> assert get_type_var_instantiation(TemplatedClass[int], T) == int
        >>> float_obj = TemplatedClass[float]()
        >>> assert get_type_var_instantiation(float_obj, T) == float

    Parameters
    ----------
    obj_or_cls:
        An instance of a templated class or the templated class itself for which the instantiated type var should
        be retrieved. If an object or type is passed which does not inherit from Generic, an error is thrown
    type_var:
        for which of the type vars that appear in the inheritance tree of the passed object or class the respective
        instantiated class should be retrieved. If a type var is passed which does not belong to any of the specified
        type vars in the inheritance tree of the passed object or class, an error is thrown

    Returns
    -------
        the instantiated class for the respective `type_var` of the templated `obj_or_cls`
    """

    generic_types, generic_type_vars = _gather_generic_types_and_vars(obj_or_cls)

    # Find the specified type_var in the gathered typevars of all superclasses and return the respective template
    # instantiation
    for generic_type_var, generic_type in zip(generic_type_vars, generic_types):
        if generic_type_var == type_var:
            return generic_type

    # If the specified type_var is not part of the inheritance tree of the passed object, an error is thrown
    raise ValueError(f"Could not find typevar `{type_var}` for `{obj_or_cls}")


def is_type_var_instantiated(obj_or_cls: Union[object, Type], type_var: TypeVar):
    """
    Returns whether the specified `type_var` is instantiated in the class definition of `obj_or_cls`.
    Type vars can stay uninstantiated when one subclasses a generic type without specifying its type variables.

    Parameters
    ----------
    obj_or_cls:
        An instance of a templated class or the templated class itself for which instantiation of `type_var` should be
        checked
    type_var:
        which type var to check

    Returns
    -------
        whether `type_var` is instantiated in the class definition of `obj_or_class`

    """
    _, generic_type_vars = _gather_generic_types_and_vars(obj_or_cls)
    return type_var in generic_type_vars


def gather_types(types: Iterable[Type]) -> Set[Type]:
    """
    Gathers all types that are listed in some way in the specified type hint.

    Examples
    --------
        gather_types(Union[TypeA, List[TypeB]]) -> {TypeA, TypeB}

    Parameters
    ----------
        types: a collection of types that will be traversed recursively

    Returns
    -------
        All types that are listed in the specified type hint
    """

    all_types = set()
    for t in types:
        # t = t if inspect.isclass(t) else type(t)  # Ensure that passed value is a class
        if is_generic_collection(t) or is_union(t):
            all_types.update(gather_types(extract_generic(t)))
        else:
            all_types.add(t)
    return all_types


def _rec_gather_generics(cls, generic_types, generic_type_vars):
    for base_class in cls.__orig_bases__:
        erased_class = get_origin(base_class)
        if erased_class == Generic:
            # Don't visit Generic superclasses as these are already implicitly handled by the subclass
            continue
        if erased_class is not None:
            # Current super class is a templated type. Hence, we can gather type vars and template instantiations
            generic_types.extend(get_args(base_class))
            generic_type_vars.extend(erased_class.__parameters__)
            _rec_gather_generics(erased_class, generic_types, generic_type_vars)


def _gather_generic_types_and_vars(obj_or_cls: Union[object, Type]):
    generic_types = None
    generic_type_vars = None
    cls_origin = get_origin(obj_or_cls)
    if cls_origin is not None:
        # The passed object is actually a generic class
        generic_types = get_args(obj_or_cls)
        generic_type_vars = cls_origin.__parameters__
    elif hasattr(obj_or_cls, "__orig_class__") and hasattr(obj_or_cls, "__parameters__"):
        # The passed object is an instance of a direct generic type, i.e., directly subclasses Generic[...]
        generic_types = get_args(obj_or_cls.__orig_class__)
        generic_type_vars = obj_or_cls.__parameters__
    elif hasattr(obj_or_cls, "__orig_bases__"):
        # The passed object is not itself a generic type, but inherits from templated types
        generic_types = []
        generic_type_vars = []

        # Gather all typevars and their associated types of all superclasses of the passed object
        _rec_gather_generics(obj_or_cls, generic_types, generic_type_vars)

    # If none of the above cases occurred, the passed object apparently was not a templated type
    assert generic_types is not None, f"Could not determine template types of `{obj_or_cls}`. Is it a generic type or instance?"
    assert generic_type_vars is not None, f"Could not determine generic type vars of `{obj_or_cls}`. Is it a generic type or instance?"

    return generic_types, generic_type_vars
