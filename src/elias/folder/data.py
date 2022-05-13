import re
from typing import Union, TypeVar, Generic, List, Optional

from silberstral import reveal_type_var

from elias.folder.folder import Folder
from elias.util import ensure_directory_exists
from elias.util.version import Version

_DataManagerType = TypeVar("_DataManagerType", bound='BaseDataManager')

DATASET_VERSION_REGEX = re.compile(r"v(\d+(?:\.\d+)*)(?:-.*)?")


class DataFolder(Folder, Generic[_DataManagerType]):
    """
    A DataFolder refers to a file system folder that contains the artefacts created by a data preprocessing stage.
    Each preprocessing run is stored in a separate sub folder.
    Research workflows often exhibit several iterations of data preprocessing which is why the preprocessing runs are
    tagged with a version.
    Names of sub folders are assumed to follow the format `v{version}-{name}`.
    Examples: "v1-first-dataset" or "v2.3.1-imagenet-with-more-dogs".

    DataFolders should be subclassed for each specific dataset type, e.g., one for ImageNet, one for MNIST, etc...
    During subclassing _DataManagerType is replaced by the actual class of the data manager that handles this dataset
    type.
    """

    _version_levels: int
    _default_bump_level: int
    _cls_data_manager: _DataManagerType

    def __init__(self,
                 location: str,
                 version_levels: int = 2,
                 default_bump_level: int = -1,
                 localize_via_run_name: bool = False):
        """
        Parameters
        ----------
            location:
                folder in which the preprocessing runs are stored
            version_levels:
                how many levels should be used for versioning, i.e., v1.2 or v1.2.3.4 etc.
            default_bump_level:
                which level should be increased when a new dataset is created
            localize_via_run_name:
                whether only the dataset name will be used for finding the corresponding single dataset folder
        """
        ensure_directory_exists(location)
        super(DataFolder, self).__init__(location)

        self._version_levels = version_levels
        self._default_bump_level = default_bump_level
        self._cls_data_manager = reveal_type_var(self, _DataManagerType)
        self._localize_via_run_name = localize_via_run_name

    def list_datasets(self) -> List[str]:
        """
        Lists all sub folders that obey the `v{version}-{name}` format.

        Returns
        -------
            a list containing the found dataset names
        """

        dataset_versions = self.list_dataset_versions()
        dataset_names = [self.get_dataset_name_by_version(version) for version in dataset_versions]
        return dataset_names
        # dataset_folders = self.ls()
        # dataset_folders = [f for f in dataset_folders if DATASET_VERSION_REGEX.match(f)]
        # return dataset_folders

    def list_dataset_versions(self) -> List[Version]:
        """
        Lists all the versions from the sub folders that obey the `v{version}-{name}` format.

        Returns
        -------
            a list containing the found datasets' versions
        """

        dataset_folders = self.ls()
        dataset_versions = []

        for folder in dataset_folders:
            p = DATASET_VERSION_REGEX.match(folder)
            if p:
                version_specifier = p.group(1)
                dataset_versions.append(Version(version_specifier))

        return sorted(dataset_versions)

    def get_dataset_name_by_version(self, dataset_version: Union[str, Version]) -> str:
        """
        Given its version, retrieves the full dataset name of a sub folder that obeys the `v{version}-{name}` format.

        Parameters
        ----------
            dataset_version: the version to search for

        Returns
        -------
            the full name of the dataset the belongs to the specified version
        """

        if isinstance(dataset_version, str) and dataset_version.startswith('v'):
            dataset_version = dataset_version[1:]
        dataset_folders = self.ls()

        for folder in dataset_folders:
            p = DATASET_VERSION_REGEX.match(folder)
            if p:
                version_specifier = p.group(1)
                if version_specifier == dataset_version:
                    return folder

        raise ValueError(f"Could not find dataset version `{dataset_version}` in folder `{self._location}`")

    def open_dataset(self, dataset_version: Union[str, Version]) -> _DataManagerType:
        """
        Navigates into the sub folder that belongs to the specified dataset version.
        A data manager of the respective type is created rooted at that folder for easy interaction with the stored
        artifacts.

        Parameters
        ----------
            dataset_version: version or full name of the dataset that shall be opened

        Returns
        -------
            a data manager to interact with the dataset files
        """

        dataset_name = self._get_full_dataset_name(dataset_version)

        # Create a new data manager of the corresponding class for the specified dataset folder
        return self._cls_data_manager.from_location(self._location,
                                                    dataset_name,
                                                    localize_via_run_name=self._localize_via_run_name)

    def create_dataset(self, name: Optional[str] = None, bump_level: Optional[int] = None) -> _DataManagerType:
        """
        Creates a new data manager for a dataset with the given name. It is automatically assigned a version that is
        higher than the existing ones.

        Parameters
        ----------
            name: name of the new dataset (without version specifier, this will be added automatically)
            bump_level:
                which level of the version should be increased. Levels are 0-indexed.
                If `None`, the default specified in the constructor is used

        Returns
        -------
            A data manager for the newly created (empty) dataset folder
        """

        bump_level = self._default_bump_level if bump_level is None else bump_level
        if len(self.list_datasets()) == 0:
            # First dataset in this folder. Initialize with 0.0.1 (or similar)
            initial_version = Version.from_zero(self._version_levels)
            initial_version.bump(bump_level)
            new_version = initial_version
        else:
            # Some datasets already exist. Find maximum version and bump
            versions = self.list_dataset_versions()
            max_version = max(versions)
            max_version.bump(bump_level)
            new_version = max_version

        dataset_name = f"v{new_version}" if name is None else f"v{new_version}-{name}"
        self.mkdir(dataset_name)

        return self._cls_data_manager.from_location(self._location,
                                                    dataset_name,
                                                    localize_via_run_name=self._localize_via_run_name)

    def remove_dataset(self, dataset_version: Union[str, Version]):
        """
        Deletes the specified dataset folder and all its content.

        Parameters
        ----------
            dataset_version: version or full name of the dataset to be deleted
        """

        dataset_name = self._get_full_dataset_name(dataset_version)
        self.rmdir(dataset_name)

    def _get_full_dataset_name(self, dataset_version: Union[str, Version]) -> str:
        dataset_version = str(dataset_version)
        if Version.is_valid(dataset_version):
            # Only version was given, need to find corresponding dataset name
            return self.get_dataset_name_by_version(dataset_version)
        else:
            # Full dataset name (including version specifier) was given
            return dataset_version
