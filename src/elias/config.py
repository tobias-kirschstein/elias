from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, asdict, fields, field, replace, is_dataclass
from enum import Enum, EnumMeta, auto
from importlib import import_module
from pydoc import locate
from typing import List, Tuple, Any, Type, get_type_hints, Generic, TypeVar, Dict, Iterator, Callable, Optional

import dacite
import numpy as np
from dacite import from_dict
from dacite.dataclasses import get_fields
from silberstral import gather_types, is_type_var_instantiated, reveal_type_var

# TODO: Implement Dict or_else() method

# =========================================================================
# Better Enum handling for persistable config objects
# =========================================================================
from elias.util.typing import class_to_module_path, module_path_to_class

_T_Enum = TypeVar('_T_Enum', bound=Enum)


class NamedEnumMeta(EnumMeta):
    """
    Enum Meta to enable instantiating enums with their name as well.
    """

    def __call__(cls: NamedEnum, value, *args, **kw):
        if isinstance(value, str):
            value = cls.from_name(value).value
        return super().__call__(value, *args, **kw)

    def __iter__(self: _T_Enum) -> Iterator[_T_Enum]:
        return super(NamedEnumMeta, self).__iter__()


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

    def __iter__(self: _T_Enum) -> Iterator[_T_Enum]:
        # Additional type hinting, as StringEnum inherits from str and PyCharm's type checker otherwise thinks
        # iterating over the enum yields strings
        return super(StringEnum, self).__iter__()


class ClassMapping(StringEnum):
    """
    Class Mapping enums serve the purpose of making persisted class names more human readable. When an
    `AbstractDataClass` is defined with a corresponding `ClassMapping`, sub class of the data class will be stored
    by their respective enum name instead of the fully qualified class name. This makes it easier for humans to specify
    subclasses of `AbstractDataClass` in a config file.
    For this to work, one simply defines a ClassMapping enum for every sub class of the `AbstractDataClass` and passes
    the ClassMapping enum as a type var to the `AbstractDataClass`.
    """

    @classmethod
    @abstractmethod
    def get_mapping(cls) -> Dict[ClassMapping, Type]:
        pass


# =========================================================================
# Actual Config class
# =========================================================================

@dataclass
class Config(ABC):

    def _serialize_enums_and_numpy(self, items: List[Tuple[str, Any]]):
        d = dict()
        config_fields = {f.name: f for f in fields(self)}
        for key, value in items:
            if isinstance(value, Enum):
                value = value.value  # Use the Enum's value as representation for the value
            elif isinstance(value, dict):
                # If a Config defines a member of type Dict it won't be unrolled by the dataclass asdict() method
                # Hence, unroll it manually here. Enums are fine for both keys and values, but they have to be
                # serialized
                value = {
                    k.value if isinstance(k, Enum) else k:
                        v.value if isinstance(v, Enum) else v
                    for k, v in value.items()}

            # TODO: Same issue as with numpy handling
            #   We can only serialize 'Type' fields if we don't check for the type annotation in the dataclass
            #   This is risky, because in an ideal scenario we would only serialize a class value if the field
            #   is actually supposed to hold a class
            elif inspect.isclass(value):  # and key in config_fields and config_fields[key].type == Type:
                # Handling for fields with type 'Type':
                # represent the Type as a module import string, e.g., np.ndarray
                value = class_to_module_path(value)

            # TODO: due to dacite we can only unpack numpy items at the outer level.
            #   dacite does iterate through nested configs of course, but here it does not give us access to the fields
            #   of the nested config...
            elif key in config_fields \
                    and (config_fields[key].type == float and (
                    isinstance(value, np.float32) or isinstance(value, np.float64))
                         or (config_fields[key].type == int and (
                            isinstance(value, np.int32) or isinstance(value, np.int64)))
                         or (config_fields[key].type == bool and isinstance(value, np.bool_))):
                # If a single valued numpy object was passed, but a built-in Python type was expected, we need to
                # extract the value from the numpy container. This is for the convenience of the user
                # Currently implemented casts:
                #  - np.float32/64 -> float
                #  - np.int32/64 -> int
                #  - np.bool_ -> bool
                value = value.item()

            d[key] = value
        return d

    def to_json(self) -> dict:
        """
        Converts this configuration dataclass into an ordinary Python dictionary that can easily be persisted as JSON.
        Special attention is given to enum members of the dataclass. As enums cannot be serialized per default, these
        are represented by their intrinsic value instead. When deserializing the stored JSON with :meth:`from_json`
        the enum values can be parsed into proper enums again thanks to the type annotation in the underlying dataclass.

        Returns
        -------
            a Python dictionary representing this dataclass
        """

        config = replace(self)

        # Python's asdict() cannot deal with defaultdict instances "TypeError: first argument must be callable or None"
        # Hence, we silently replace defaultdicts with regular dicts here
        def _cast_defaultdict_rec(inner_config: Config):
            for field in fields(inner_config):
                value = getattr(inner_config, field.name)
                if isinstance(value, defaultdict):
                    setattr(inner_config, field.name, dict(value))
                elif is_dataclass(value):
                    _cast_defaultdict_rec(value)

        _cast_defaultdict_rec(config)

        return asdict(config, dict_factory=config._serialize_enums_and_numpy)

    def __post_init__(self):
        """
        Overrides the post initialization hook of Python's dataclasses.
        This default implementation scans for fields in the dataclass that have an enum type hint.
        Any of these attributes that are not proper enums of their type will then be automatically instantiated.
        This comes handy when initializing a Config from command line values as these are often only strings hinting
        at the enum member.
        """

        # Double check that self is actually a dataclass. Otherwise, one will potentially get weird bugs downstream
        field_names = {field.name for field in get_fields(type(self))}
        assert all([k in field_names for k in get_type_hints(type(self)).keys()]), \
            f"Not all hinted types in `{self}` appear in its dataclass field list. Is it a dataclass?"

        casts = self.__class__._define_casts()
        # TODO: Can be that we have to use get_type_hints() instead of fields() here, as fields does not contain
        #  the actual classes when from __future__ import annotations is used
        for field in fields(self):
            # Only check visible fields (not hidden via field(init=False))
            if hasattr(self, field.name):
                field_value = getattr(self, field.name)
                if field.type in casts and not isinstance(field_value, field.type):
                    setattr(self, field.name, field.type(field_value))

        # if hasattr(self, "_enable_backward_compatibility") and self._enable_backward_compatibility:
        #     self._backward_compatibility()
        #
        # for field in fields(self):
        #     if "deprecated" in field.metadata and field.metadata["deprecated"] and hasattr(self, field.name):
        #         # We expect that a field that is marked as deprecated has already been dealt with in
        #         # _backward_compatibility()
        #         delattr(self, field.name)
        #
        #     if "required" in field.metadata and field.metadata["required"]:
        #         if getattr(self, field.name) is None:
        #             raise ValueError(f"Field {field.name} is required!")

    @classmethod
    def _backward_compatibility(cls, loaded_config: Dict):
        """
        Allows coping with Configs that change over time by manually altering the `loaded_config` to adhere to the
        current version of the Config.

        Parameters
        ----------
            loaded_config: the config values that have been loaded from a (potentially) outdated config file.

        """

        # Recursively go through all fields and give them the possibility to apply backward compatibility
        for f in fields(cls):
            # In case, a field has a Union/List etc. type we need to check all of them
            # TODO: What if we have Union[A, B] and the _backward_compatibility() methods of A and B disagree?
            possible_types = gather_types([f.type])
            for t in possible_types:
                t = t if inspect.isclass(t) else type(t)
                if issubclass(t, Config) and f.name in loaded_config:
                    # If a field is a Config Type, apply its backward compatibility method
                    # TODO: How do we want to handle cases like List[SomeConfig]?
                    #   Currently, the loaded_config[f.name] will be the list of items and the config class will
                    #   have to deal with unpacking itself.
                    #   This can be super complex like Tuple[SomeConfig, List[Union[SomeConfig2, SomeConfig3]]]...
                    sub_dict = loaded_config[f.name]

                    # Only traverse further if dictionary is not None.
                    # After all, what should a dataclass do in its backward_compatibility() method if the passed dict
                    # is None?
                    if sub_dict is not None:
                        t._backward_compatibility(sub_dict)

    @classmethod
    def _define_casts(cls) -> List[Type]:
        """
        Lists all the types of dataclass attributes that are enums and thus should be casted to their proper types
        if they not yet belong to it.

        Returns
        -------
            a list of types for which explicit conversion is initiated whenever this Config dataclass is instantiated
        """

        # TODO: Can we automatically cast 'None' to None?

        casts = []
        field_types = get_type_hints(cls).values()
        # Find all mentioned types in the dataclass definition (even those mentioned as generics)
        for field_type in gather_types(field_types):
            if inspect.isclass(field_type):
                # Automatically cast to enums and custom types that were listed in config fields
                if issubclass(field_type, Enum):  # or not inspect.isbuiltin(field_type):
                    casts.append(field_type)
        casts.append(tuple)  # Tuples are stored as [] lists in JSON. Cast them back to tuple here

        return casts

    # TODO: rename. It doesn't make sense that this method is called from_json
    @classmethod
    def from_json(cls,
                  json_config: dict,
                  type_hooks: Optional[Dict[Type, Callable[[Any], Any]]] = None) -> Config:
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
        type_hooks: Dict[Type, Callable[[Any], Any]]]
            type hooks can be used to guide the deserialization process.
            For example, a complicated data structure may be serialized as a series of lists, but in the loaded config
            one may want to hold the data structure and not the serialized version of it.
            In this case, a type hook defines the mapping from serialized -> data structure
            Per-default, a type hook that maps series of lists back to numpy arrays is already added:
            {
                np.ndarray: lambda array_values: np.asarray(array_values)
            }

        Returns
        -------
            This dataclass with all the values from :attr:`json_config` filled in. Enum attributes are explicitly
            converted.

        """

        abstract_dataclasses = []
        data_sub_class_types = []

        for field_type in get_type_hints(cls).values():
            field_type = field_type if inspect.isclass(field_type) else type(field_type)
            if issubclass(field_type, AbstractDataclass):
                abstract_dataclasses.append(field_type)
                if is_type_var_instantiated(field_type, DataSubclassType):
                    data_sub_class_types.append(reveal_type_var(field_type, DataSubclassType))
                else:
                    data_sub_class_types.append(None)

        def instantiate_adc_with_sub_class(abstract_dataclass_values: dict, data_sub_class_type):
            # Instantiate the abstract data class field within the data class with its respective subclass as hinted
            # by the attribute 'type'
            if data_sub_class_type is None:
                # AbstractDataClass does not have a corresponding enum class mapping -> interpret value as
                # fully qualified class name
                sub_class = locate(abstract_dataclass_values['type'])
                assert sub_class is not None, f"Could not locate class {abstract_dataclass_values['type']}. " \
                                              f"Is it globally accessible, i.e., not defined in local scope?"
            else:
                # Use the Class Mapping as a lookup to get the actual sub class that should be instantiated
                class_mapping = data_sub_class_type.get_mapping()
                assert abstract_dataclass_values['type'].upper() in class_mapping, \
                    f"Could not find specified type `{abstract_dataclass_values['type']}` " \
                    f"in class mapping of {data_sub_class_type}"
                sub_class = class_mapping[abstract_dataclass_values['type']]

            # Delete the type attribute from the JSON input as it is implicitly defined
            del abstract_dataclass_values['type']

            return sub_class.from_json(abstract_dataclass_values)

        all_type_hooks = {
            # Use Lambda closure (i=i) to ensure data_sub_class_type is copied for each lambda
            abstract_dataclass:
                lambda abstract_dataclass_values, data_sub_class_type=data_sub_class_type:
                instantiate_adc_with_sub_class(abstract_dataclass_values, data_sub_class_type)
            for abstract_dataclass, data_sub_class_type
            in zip(abstract_dataclasses, data_sub_class_types)}

        # Numpy arrays are serialized as lists. Cast them back to np array here
        all_type_hooks[np.ndarray] = lambda array_values: np.asarray(array_values)
        all_type_hooks[Type] = module_path_to_class

        if type_hooks is not None:
            # Add use-defined type hooks
            all_type_hooks.update(type_hooks)

        # Register type hooks to replace every single AbstractDataClass with the respective subclass hinted by the
        # 'type' attribute
        dacite_config = dacite.Config(
            cast=cls._define_casts(),
            type_hooks=all_type_hooks,
            strict=False)

        # backward_cls = type(cls.__name__, cls.__bases__, dict(cls.__dict__))
        #
        # def backward_compatibility_new(cls_new, *args, **kwargs):
        #     obj = super(Config, cls).__new__(cls)
        #     obj._enable_backward_compatibility = True
        #     # For some reason, __init__ isn't called anymore if __new__ is overridden. So manually call it here
        #     obj.__init__(*args, **kwargs)
        #     return obj
        #
        # backward_cls.__new__ = backward_compatibility_new
        # return from_dict(backward_cls, json_config, config=dacite_config)
        cls._backward_compatibility(json_config)
        try:
            config = from_dict(cls, json_config, config=dacite_config)
        except IndexError as e:
            referenced_types = gather_types(cls)
            if Type in referenced_types:
                # In case a field has type "Type" dacite unfortunately throws an unecessary error
                # IndexError: tuple index out of range
                # happening in types.py:129
                dacite_config.check_types = False
                config = from_dict(cls, json_config, config=dacite_config)
            else:
                raise e

        return config

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


# =========================================================================
# AbstractDataClass to allow describing inheritance structures in dataclass fields
# =========================================================================


DataSubclassType = TypeVar("DataSubclassType", bound=ClassMapping)


@dataclass
class AbstractDataclass(Generic[DataSubclassType], Config):
    """
    An AbstractDataclass allows employing (human-configurable) sub-class structures in Config dataclasses.
    Per default, it is not possible to use subclasses of a common superclass as attributes in a dataclass, as during
    deserialization one cannot know which of the subclasses was serialized.
    This is solved by inserting a 'type' attribute for sub classes which is filled during instantiation and which is
    persisted. This 'type' attribute then enables deserializing the config values as it specifies the corresponding
    class to create.
    To ensure that this 'type' attribute is filled, the common superclass has to inherit from `AbstractDataClass`.
    Optionally, the superclass can specify a `ClassMapping` enum to map the persisted class names of its sub classes
    to more human-readable (and -editable) strings.

    Examples
    --------

        >>> class MyClassMapping(ClassMapping):
        >>>     CLASS_A = auto()
        >>>     CLASS_B = auto()
        >>>
        >>>     @classmethod
        >>>     def get_mapping(cls) -> Dict[ClassMapping, Type]:
        >>>         return {cls.CLASS_A: MySubClassA, cls.CLASS_B: MySubClassB}
        >>>
        >>> @dataclass
        >>> class MySuperClass(AbstractDataclass[MyClassMapping]):
        >>>     super_class_attribute: int
        >>>
        >>> @dataclass
        >>> class MySubClassA(MySuperClass):
        >>>     a1: float
        >>>
        >>> @dataclass
        >>> class MySubClassB(MySuperClass):
        >>>     b1: str
        >>>     b2: int
        >>>
        >>> @dataclass
        >>> class MyConfig(Config):
        >>>     config_value: int
        >>>     my_object: MySuperClass
        >>>
        >>> my_config = MyConfig(1, MySubClassB("b1", 2))
        >>> assert MyConfig.from_json(my_config.to_json()) == my_config
    """
    type: str = field(init=False, repr=False)

    def __new__(cls, *args, **kwargs):
        if cls == AbstractDataclass or cls.__bases__[0] == AbstractDataclass:
            raise TypeError("Cannot instantiate abstract class.")
        return super().__new__(cls)

    def __post_init__(self):
        if is_type_var_instantiated(self, DataSubclassType):
            # This AbstractDataClass has a corresponding class mapping enum. Use the respective enum name
            # for this instance as 'type' attribute
            data_sub_class_enum: ClassMapping = reveal_type_var(self, DataSubclassType)
            sub_class_mapping = data_sub_class_enum.get_mapping()
            sub_class = None
            for sub_class_name, sub_class_type in sub_class_mapping.items():
                if sub_class_type == type(self):
                    sub_class = sub_class_name

            assert sub_class is not None, f"Could not find {type(self)} in mapping {sub_class_mapping} of {data_sub_class_enum}"
        else:
            # No ClassMapping defined -> use fully qualified name of this instance as 'type' attribute
            cls = type(self)
            module = cls.__module__
            if module == '__builtin__':
                sub_class = cls.__qualname__  # avoid outputs like '__builtin__.str'
            else:
                sub_class = f"{cls.__module__}.{cls.__qualname__}"

        self.type = sub_class

        super(AbstractDataclass, self).__post_init__()


def implicit(default: Any = None):
    """
    Hints at that the specified field is to be initialized implicitly by other values and just exists for convenience.
    Implicit fields cannot be directly specified via the constructor. Instead they have to be set after the config
    has already been created. However, implicit values will also be listed when :meth:`to_json()` is called and they
    can be directly set via :meth:`from_json()`, i.e., they can be loaded from persisted files.
    Use Optional[T] if no default is specified as implicit values will default to None in that case.

    Parameters
    ----------
        default: Any
            The default value that will be used in case the implicit attribute is never defined

    Returns
    -------
        a dataclass type hint that this attribute is implicitly defined by other values

    Examples
    --------
        >>> @dataclass
        >>> class MyConfig(Config):
        >>>     a: int
        >>>     b: int = implicit(0)
        >>>     c: Optional[int] = implicit()
    """

    return field(init=False, default=default)
