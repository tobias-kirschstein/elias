from dataclasses import dataclass
from enum import auto, Enum
from typing import Dict, Type, List, Tuple, Optional, Union
from unittest import TestCase
import numpy as np
from testfixtures import TempDirectory

from elias.config import AbstractDataclass, Config, ClassMapping, StringEnum
from elias.util import save_json, load_json


# TODO: We may have an issue with forward references (types denoted with 'SomeType')
#   These have to be evaluated to know which class they refer to
#   Evaluation is necessary in _backward_compatibility()
#   However, evaluation of a forward reference is not straight forward. Maybe print a warning in this case?
#   Not even sure, what will break in this case
@dataclass
class InnerConfig(Config):
    b: str


@dataclass
class OptionalNestedConfig(Config):
    i: int
    nested: Optional[InnerConfig]


class ConfigTest(TestCase):
    class SuperClassType(ClassMapping):
        A = auto()
        B = auto()

        @classmethod
        def get_mapping(cls) -> Dict[ClassMapping, Type]:
            return {cls.A: ConfigTest.AWithMapping,
                    cls.B: ConfigTest.BWithMapping}

    class SuperClassWithMapping(AbstractDataclass[SuperClassType]):
        pass

    @dataclass
    class AWithMapping(SuperClassWithMapping):
        a: int

    @dataclass
    class BWithMapping(SuperClassWithMapping):
        b1: str
        b2: float

    class SuperClassWithoutMapping(AbstractDataclass):
        pass

    @dataclass
    class AWithoutMapping(SuperClassWithoutMapping):
        a: int

    @dataclass
    class BWithoutMapping(SuperClassWithoutMapping):
        b1: str
        b2: float

    @dataclass
    class ConfigWithNumpyArray(Config):
        array: np.ndarray
        arrays: List[np.ndarray]

    @dataclass
    class ConfigWithNumbers(Config):
        i: int
        f: float
        b: bool

    @dataclass
    class NestedConfigWithNumbers(Config):
        i2: int
        f: float
        nested: 'ConfigTest.ConfigWithNumbers'

    @dataclass
    class ConfigWithTuple(Config):
        some_tuple: Tuple[float, int, str]

    class ComplicatedType(np.ndarray):

        def __new__(cls, values: Union[List, np.ndarray], *args, **kwargs):
            values = np.asarray(values)
            pose = super().__new__(cls, values.shape, dtype=np.float32)
            pose[:] = values
            return pose

    @dataclass
    class TypeHookConfig(Config):
        data_structure: 'ConfigTest.ComplicatedType'

    @dataclass
    class ConfigWithType(Config):
        my_type: Type

        nested: Optional['ConfigTest.ConfigWithType'] = None

    @dataclass
    class ConfigWithWrappedType(Config):
        my_config: 'ConfigTest.ConfigWithType'

    @dataclass
    class ConfigWithUnion(Config):
        numbers_or_tuple: Union['ConfigTest.ConfigWithNumbers', 'ConfigTest.ConfigWithTuple']

    # -------------------------------------------------------------------------
    # Begin Tests
    # -------------------------------------------------------------------------

    @staticmethod
    def _serialize_then_unserialize(cls: Type[Config], c: Config) -> Config:
        c_serialized = c.to_json()
        with TempDirectory() as d:
            save_json(c_serialized, d.path)
            c_loaded = load_json(d.path)
        c_reconstructed = cls.from_json(c_loaded)
        return c_reconstructed

    def test_abstract_dataclass_with_mapping(self):
        @dataclass
        class TestConfigWithMapping(Config):
            test: ConfigTest.SuperClassWithMapping
            other: int

        a_test = ConfigTest.AWithMapping(1)
        b_test = ConfigTest.BWithMapping("b", 1.1)

        ConfigTest.AWithMapping.from_json(a_test.to_json())
        ConfigTest.BWithMapping.from_json(b_test.to_json())

        tc = TestConfigWithMapping.from_json(TestConfigWithMapping(a_test, 2).to_json())
        self.assertEqual(tc.test, a_test)
        self.assertEqual(tc.to_json()['test']['type'], ConfigTest.SuperClassType.A.name)

        tc = TestConfigWithMapping.from_json(TestConfigWithMapping(b_test, 2).to_json())
        self.assertEqual(tc.test, b_test)
        self.assertEqual(tc.to_json()['test']['type'], ConfigTest.SuperClassType.B.name)

        print(isinstance(ConfigTest.SuperClassType.A, ClassMapping))

    def test_abstract_dataclass_without_mapping(self):
        @dataclass
        class TestConfigWithoutMapping(Config):
            test: ConfigTest.SuperClassWithoutMapping
            other: int

        a_test = ConfigTest.AWithoutMapping(1)
        b_test = ConfigTest.BWithoutMapping("b", 1.1)

        ConfigTest.AWithoutMapping.from_json(a_test.to_json())
        ConfigTest.BWithoutMapping.from_json(b_test.to_json())

        tc = TestConfigWithoutMapping.from_json(TestConfigWithoutMapping(a_test, 2).to_json())
        self.assertEqual(tc.test, a_test)
        self.assertEqual(tc.to_json()['test']['type'],
                         f"{ConfigTest.AWithoutMapping.__module__}.{ConfigTest.AWithoutMapping.__qualname__}")

        tc = TestConfigWithoutMapping.from_json(TestConfigWithoutMapping(b_test, 2).to_json())
        self.assertEqual(tc.test, b_test)
        self.assertEqual(tc.to_json()['test']['type'],
                         f"{ConfigTest.BWithoutMapping.__module__}.{ConfigTest.BWithoutMapping.__qualname__}")

    # def test_deprecated_attribute(self):
    #     @dataclass
    #     class OldTestConfig(Config):
    #         old: int = 3
    #
    #     @dataclass
    #     class TestConfig(Config):
    #         regular: int = backward_compatible()
    #         old: int = deprecated(3)
    #
    #         def _backward_compatibility(self):
    #             print("backward_compatibility")
    #             super(TestConfig, self)._backward_compatibility()
    #
    #             if self.old != 3 and self.regular is None:
    #                 self.regular = self.old
    #
    #     old_config = OldTestConfig(7)
    #     old_json = old_config.to_json()
    #     new_config = TestConfig.from_json(old_json)
    #
    #     self.assertNotIn('old', new_config.__dict__)
    #     self.assertEqual(new_config.regular, old_config.old)
    #
    #     new_config = TestConfig(regular=5)
    #     self.assertEqual(new_config.regular, 5)
    #     self.assertNotIn('old', new_config.__dict__)
    #     print(new_config.to_json())
    #
    #     new_config = TestConfig()

    def test_config_dict_with_enums(self):
        class VanillaEnum(Enum):
            A = auto()
            B = auto()

        class TestStringEnum(StringEnum):
            A = auto()
            B = auto()

        @dataclass
        class ConfigWithEnumDict(Config):
            d_regular_key: Dict[VanillaEnum, int]
            d_regular_value: Dict[int, VanillaEnum]
            d_regular_key_value: Dict[VanillaEnum, VanillaEnum]

            d_string_key: Dict[TestStringEnum, int]
            d_string_value: Dict[int, TestStringEnum]
            d_string_key_value: Dict[TestStringEnum, TestStringEnum]

            d_regular_string: Dict[VanillaEnum, TestStringEnum]
            d_string_regular: Dict[TestStringEnum, VanillaEnum]

        d_regular_key = {VanillaEnum.A: 1, VanillaEnum.B: 2}
        d_regular_value = {1: VanillaEnum.A, 2: VanillaEnum.B}
        d_regular_key_value = {VanillaEnum.A: VanillaEnum.A, VanillaEnum.B: VanillaEnum.A}

        d_string_key = {TestStringEnum.A: 1, TestStringEnum.B: 2}
        d_string_value = {1: TestStringEnum.A, 2: TestStringEnum.B}
        d_string_key_value = {TestStringEnum.A: TestStringEnum.A, TestStringEnum.B: TestStringEnum.A}

        d_regular_string = {VanillaEnum.A: TestStringEnum.A, VanillaEnum.B: TestStringEnum.A}
        d_string_regular = {TestStringEnum.A: VanillaEnum.A, TestStringEnum.B: VanillaEnum.A}

        c = ConfigWithEnumDict(d_regular_key, d_regular_value, d_regular_key_value,
                               d_string_key, d_string_value, d_string_key_value,
                               d_regular_string, d_string_regular)

        c_serialized = c.to_json()
        for serialized_enum_dict in c_serialized.values():
            for key, value in serialized_enum_dict.items():
                # All keys and values should have been converted to their str or int representations
                self.assertTrue(isinstance(key, str) or isinstance(key, int))
                self.assertTrue(isinstance(value, str) or isinstance(value, int))

        c_reconstructed = c.from_json(c_serialized)
        self.assertEqual(c_reconstructed, c)

    def test_config_with_np_array(self):
        array = np.eye(2)
        arrays = [np.eye(3), np.ones(4), np.array([[1, 2], [3, 4]])]
        config = self.ConfigWithNumpyArray(array, arrays)
        c_serialized = config.to_json()

        with TempDirectory() as d:
            save_json(c_serialized, d.path)
            c_loaded = load_json(d.path)

        c_reconstructed = self.ConfigWithNumpyArray.from_json(c_loaded)
        self.assertTrue((config.array == c_reconstructed.array).all())
        self.assertTrue((config.arrays[0] == c_reconstructed.arrays[0]).all())
        self.assertTrue((config.arrays[1] == c_reconstructed.arrays[1]).all())
        self.assertTrue((config.arrays[2] == c_reconstructed.arrays[2]).all())

    def test_config_tuple(self):
        some_tuple = (3.14, 5, "hi")
        c = self.ConfigWithTuple(some_tuple)
        # TODO: This test shows why we need to get away from dacite.
        #   Dacite does not maps the type hints from the Config one to one to the loaded dict
        #   Instead, it tries to apply all configured casts to every field as long as the cast field is a subclass
        c_reconstructed = self._serialize_then_unserialize(self.ConfigWithTuple, c)
        self.assertEqual(c, c_reconstructed)

    def test_config_np_items(self):
        config = self.ConfigWithNumbers(np.array([1], dtype=np.int64).max(), np.array([1], dtype=np.float64).max(),
                                        True)
        config_json = config.to_json()
        self.assertEqual(type(config_json['i']), int)
        self.assertEqual(type(config_json['f']), float)

        config = self.ConfigWithNumbers(np.array([1], dtype=np.int32).max(), np.array([1], dtype=np.float32).max(),
                                        True)
        config_json = config.to_json()
        self.assertEqual(type(config_json['i']), int)
        self.assertEqual(type(config_json['f']), float)

        config = self.ConfigWithNumbers(1, 1, np.array([True]).all())
        config_json = config.to_json()
        self.assertEqual(type(config_json['b']), bool)

        nested_config = self.ConfigWithNumbers(
            np.array([1], dtype=np.int32).max(),
            np.array([1], dtype=np.float32).max(),
            np.array([True]).all())
        config = self.NestedConfigWithNumbers(np.array([1], dtype=np.int32).max(), 3.14, nested_config)
        config_json = config.to_json()
        self.assertEqual(type(config_json['i2']), int)
        # self.assertEqual(type(config_json['nested']['b']), bool)
        # TODO: This is another test that shows why we need to get away from dacite
        #   Numpy item unpacking only works on the outer level, but not in nested configs

    def test_config_backward_compatibility_none_field(self):
        config = OptionalNestedConfig(2, None)
        config_json = config.to_json()
        OptionalNestedConfig.from_json(config_json)

    def test_type_hooks(self):
        data_structure = ConfigTest.ComplicatedType([1, 2, 3])
        type_hook_config = ConfigTest.TypeHookConfig(data_structure)
        type_hook_config_json = type_hook_config.to_json()

        with TempDirectory() as d:
            save_json(type_hook_config_json, d.path)
            type_hook_config_loaded = load_json(d.path)

        type_hook_config_reconstructed = ConfigTest.TypeHookConfig.from_json(
            type_hook_config_loaded,
            type_hooks={ConfigTest.ComplicatedType: lambda value: ConfigTest.ComplicatedType(value)}
        )

        assert (type_hook_config_reconstructed.data_structure == type_hook_config.data_structure).all()

    def test_config_with_type(self):
        config = ConfigTest.ConfigWithType(np.ndarray, nested=ConfigTest.ConfigWithType(str))
        config_json = config.to_json()
        with TempDirectory() as d:
            save_json(config_json, d.path)
            config_json_loaded = load_json(d.path)

        config_reconstructed = ConfigTest.ConfigWithType.from_json(config_json_loaded)
        self.assertEqual(config_reconstructed, config)

        config = ConfigTest.ConfigWithWrappedType(config)
        config_json = config.to_json()
        with TempDirectory() as d:
            save_json(config_json, d.path)
            config_json_loaded = load_json(d.path)

        config_reconstructed = ConfigTest.ConfigWithWrappedType.from_json(config_json_loaded)
        self.assertEqual(config_reconstructed, config)

    def test_config_with_union(self):
        config_with_numbers = ConfigTest.ConfigWithNumbers(1, 2.3, True)
        config_with_tuple = ConfigTest.ConfigWithTuple((2.5, 3, "test"))
        config_1 = ConfigTest.ConfigWithUnion(config_with_numbers)
        config_2 = ConfigTest.ConfigWithUnion(config_with_tuple)

        config_json_1 = config_1.to_json()
        config_json_2 = config_2.to_json()
        with TempDirectory() as d:
            save_json(config_json_1, f"{d.path}/config_1.json")
            save_json(config_json_2, f"{d.path}/config_2.json")
            config_json_1_loaded = load_json(f"{d.path}/config_1.json")
            config_json_2_loaded = load_json(f"{d.path}/config_2.json")

        config_1_reconstructed = ConfigTest.ConfigWithUnion.from_json(config_json_1_loaded)
        config_2_reconstructed = ConfigTest.ConfigWithUnion.from_json(config_json_2_loaded)
        self.assertEqual(config_1_reconstructed, config_1)
        self.assertEqual(type(config_1_reconstructed.numbers_or_tuple), ConfigTest.ConfigWithNumbers)
        self.assertEqual(config_2_reconstructed, config_2)
        self.assertEqual(type(config_2_reconstructed.numbers_or_tuple), ConfigTest.ConfigWithTuple)