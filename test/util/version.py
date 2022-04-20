from unittest import TestCase

from elias.util.version import Version


class VersionTest(TestCase):

    def test_version_from_str(self):
        version = Version("0.1")
        self.assertEqual(version.get_n_levels(), 2)
        self.assertEqual(version.get_level(0), 0)
        self.assertEqual(version.get_level(1), 1)

        version = Version("1.23.456")
        self.assertEqual(version.get_n_levels(), 3)
        self.assertEqual(version.get_level(0), 1)
        self.assertEqual(version.get_level(1), 23)
        self.assertEqual(version.get_level(2), 456)

        version = Version("99")
        self.assertEqual(version.get_n_levels(), 1)
        self.assertEqual(version.get_level(0), 99)

    def test_version_from_ints(self):
        version = Version(1, 0, 0)
        self.assertEqual(str(version), "1.0.0")

        version = Version(0, 0)
        self.assertEqual(str(version), "0.0")

    def test_version_fail(self):
        with self.assertRaises(ValueError):
            Version("a")

        with self.assertRaises(ValueError):
            Version("-1.1")

        with self.assertRaises(ValueError):
            Version("2.3.")

        with self.assertRaises(ValueError):
            Version("1..1")

        with self.assertRaises(ValueError):
            Version(".7")

        with self.assertRaises(ValueError):
            Version(1, "2")

        with self.assertRaises(ValueError):
            Version("1", "2")

        with self.assertRaises(ValueError):
            Version()

    def test_bump_version(self):
        version = Version("1.2.3")
        version.bump(0)
        self.assertEqual(version.get_level(0), 2)
        self.assertEqual(version.get_n_levels(), 3)
        self.assertEqual(version, "2.0.0")

        version = Version("1.2.3")
        version.bump(1)
        self.assertEqual(version, "1.3.0")

        version = Version("1.2.3")
        version.bump(2)
        self.assertEqual(version, "1.2.4")

        version = Version("0.9")
        version.bump(1)
        self.assertEqual(version, "0.10")

        version = Version("1.2.3.4")
        version.bump(-1)
        self.assertEqual(version, "1.2.3.5")

        version = Version("1.2.3.4")
        version.bump(-2)
        self.assertEqual(version, "1.2.4.0")

        version = Version("1.2.3.4")
        version.bump(-3)
        self.assertEqual(version, "1.3.0.0")

        version = Version("1.2.3.4")
        version.bump(-4)
        self.assertEqual(version, "2.0.0.0")

        with self.assertRaises(AssertionError):
            version = Version("1.2.3")
            version.bump(3)

        with self.assertRaises(TypeError):
            version = Version("1.2.3")
            version.bump("asdf")

    def test_version_zero(self):
        version = Version.from_zero(1)
        self.assertEqual(version, "0")

        version = Version.from_zero(3)
        self.assertEqual(version, "0.0.0")

        with self.assertRaises(ValueError):
            version = Version.from_zero(0)

        with self.assertRaises(ValueError):
            version = Version.from_zero(-2)
