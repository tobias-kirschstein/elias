from dataclasses import dataclass
from enum import auto, Enum
from typing import Dict, Type
from unittest import TestCase

from elias.config import AbstractDataclass, Config, ClassMapping, StringEnum


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
