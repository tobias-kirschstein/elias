import unittest
from pathlib import Path
from typing import Iterator

from testfixtures import TempDirectory

from elias.manager.data import BaseDataManager, _SampleType
from elias.util.version import Version
from elias.folder import DataFolder


class TestDataManager(BaseDataManager[None, None, None]):

    def __init__(self, location: str):
        super(TestDataManager, self).__init__(location, "dataset-$")

    @classmethod
    def from_location(cls, location: str) -> 'BaseDataManager':
        return TestDataManager(location)

    def _save(self, data: object, **kwargs):
        pass

    def __iter__(self) -> Iterator[_SampleType]:
        pass


class TestDataManagerWithoutFromLocation(BaseDataManager[None, None, None]):
    def __init__(self, folder: str, language: str, partition: str):
        super(TestDataManagerWithoutFromLocation, self).__init__(f"{folder}/{language}/{partition}")

    def save(self, data: object, **kwargs):
        pass

    def __iter__(self) -> Iterator[_SampleType]:
        pass


class TestDataFolder(DataFolder[TestDataManager]):

    def __init__(self, tmp_directory):
        super().__init__(f"{tmp_directory.path}/test_data_folder")


class TestDataFolderWithoutFromLocation(DataFolder[TestDataManagerWithoutFromLocation]):

    def __init__(self, tmp_directory):
        super().__init__(f"{tmp_directory.path}/test_data_folder")


class DataFolderTest(unittest.TestCase):

    def test_data_folder(self):
        TempDirectory.cleanup_all()
        with TempDirectory() as d:
            d.makedir("test_data_folder")

            data_folder = TestDataFolder(d)
            self.assertEqual(data_folder.list_datasets(), [])
            self.assertEqual(data_folder.list_dataset_versions(), [])

            data_folder.create_dataset("first-dataset")
            self.assertTrue(Path(f"{d.path}/test_data_folder/v0.1-first-dataset").exists())

            data_folder.create_dataset("second-dataset", bump_level=0)
            self.assertTrue(Path(f"{d.path}/test_data_folder/v1.0-second-dataset").exists())

            self.assertEqual(len(data_folder.list_datasets()), 2)
            self.assertEqual(len(data_folder.list_dataset_versions()), 2)

            data_folder.open_dataset("v0.1-first-dataset")
            data_folder.open_dataset("v0.1")
            data_folder.open_dataset("0.1")
            data_folder.open_dataset("1.0")
            data_folder.open_dataset(Version("1.0"))

            data_folder.remove_dataset("0.1")
            self.assertEqual(len(data_folder.list_datasets()), 1)
            self.assertEqual(len(data_folder.list_dataset_versions()), 1)

            data_folder.create_dataset("third-dataset")
            self.assertTrue(Path(f"{d.path}/test_data_folder/v1.1-third-dataset").exists())

            data_folder.remove_dataset("v1.0-second-dataset")
            self.assertEqual(len(data_folder.list_datasets()), 1)
            self.assertEqual(len(data_folder.list_dataset_versions()), 1)

            data_folder.remove_dataset("v1.1-third-dataset")
            self.assertEqual(len(data_folder.list_datasets()), 0)
            self.assertEqual(len(data_folder.list_dataset_versions()), 0)

    def test_data_folder_only_versions(self):
        TempDirectory.cleanup_all()
        with TempDirectory() as d:
            d.makedir("test_data_folder")
            data_folder = TestDataFolder(d)

            data_folder.create_dataset(bump_level=0)
            self.assertTrue(Path(f"{d.path}/test_data_folder/v1.0").exists())

            data_folder.create_dataset(bump_level=1)
            self.assertTrue(Path(f"{d.path}/test_data_folder/v1.1").exists())

    def test_data_folder_error(self):
        # If the __init__() method of a DataManager was overriden and contains different parameters than just the
        # data location, then DataFolder's create_dataset() and open_dataset() should fail
        TempDirectory.cleanup_all()
        with TempDirectory() as d:
            d.makedir("test_data_folder")
            d.makedir("test_data_folder/v1.0-test")

            data_folder = TestDataFolderWithoutFromLocation(d)
            with self.assertRaises(NotImplementedError):
                dataset = data_folder.create_dataset("test-dataset")

            with self.assertRaises(NotImplementedError):
                data_folder.open_dataset("v1.0")