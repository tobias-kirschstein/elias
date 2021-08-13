from dataclasses import dataclass
from enum import auto
from typing import Dict, Type
from unittest import TestCase

from elias.config import AbstractDataclass, Config, ClassMapping


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

