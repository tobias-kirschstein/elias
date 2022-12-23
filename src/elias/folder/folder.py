import re
from os import mkdir
from pathlib import Path
from shutil import rmtree
from typing import List, Union, Tuple, Optional

from elias.util import ensure_directory_exists


# TODO: Allow having leading zeros for $
class Folder:
    _location: str

    def __init__(self, location: str, create_if_not_exists: bool = False):
        if create_if_not_exists:
            ensure_directory_exists(location)
        # else:
        #     assert Path(f"{location}").is_dir(), \
        #         f"Could not find directory '{location}'. Is the location correct?"

        self._location = location

    def cd(self, sub_folder: str, inplace: bool = False) -> 'Folder':
        """
        Switches to the specified sub folder.

        Parameters
        ----------
            sub_folder:
                where to cd to
            inplace:
                whether the current folder object should be altered or a new one should be created

        Returns
        -------
            A new folder object at the specified location
        """

        resolved_sub_folder_path = str(Path(f"{self._location}/{sub_folder}").resolve())
        if inplace:
            self._location = resolved_sub_folder_path
            return self
        else:
            return self.__init__(resolved_sub_folder_path)

    def ls(self, name_format: Optional[str] = None) -> List[str]:
        if name_format is None:
            return [p.name for p in Path(self._location).iterdir()]
        else:
            return self.list_file_numbering(name_format, return_only_file_names=True)

    def mkdir(self, folder_name: str):
        mkdir(f"{self._location}/{folder_name}")

    def rmdir(self, folder_name: str):
        rmtree(f"{self._location}/{folder_name}")

    def get_location(self) -> str:
        return self._location

    def file_exists(self, file_name: str) -> bool:
        return Path(f"{self._location}/{file_name}").exists()

    def list_file_numbering(self,
                            name_format: str,
                            return_only_numbering: bool = False,
                            return_only_file_names: bool = False) -> List[Union[Tuple[int, str], int, str]]:
        r"""
        Finds all (numbered) files/folder in the specified directory that match the name_format and returns them in
        sorted fashion according to the number in the file/folder name.
        The numbering of the file/folder will also be returned.
        The passed name_format is expected to have exactly one occurrence of `$` which controls where in the file/folder
        name the number should appear. Additionally, `*` is supported as a wildcard similar to glob.
        This method solves the problem of numbers in file/folder names not being treated as "numbers" by the OS but
        rather as strings, i.e., 2-apple will appear after 10-banana although 2 < 10.

        Examples
        --------
            Given Folder structure:

            root
             ├── analysis-batch-norm-100-lambda-10
             ├── analysis-batch-norm-50-lambda-9
             ├── epoch-11.ckpt
             ├── epoch--1.ckpt
             ├── P2P-10
             └── P2P-9

             >>> folder = Folder('root')
             >>> folder.list_file_numbering('P2P-$', return_only_file_names=True)
             >>> >>> ['P2P-9', 'P2P-10']

             >>> folder.list_file_numbering('epoch-$.ckpt')
             >>> >>> [(-1, 'epoch--1.ckpt'), (11, 'epoch-11.ckpt')]

             >>> folder.list_file_numbering('analysis-*-$')
             >>> >>> [(9, 'analysis-batch-norm-50-lambda-9'), (10, 'analysis-batch-norm-100-lambda-10')]

        Parameters
        ----------
            name_format:
                specifies which file/folder names should be filtered and where in their name the numbering occurs.
                Two types of format specifiers are supported:
                    $: indicates where in the file name the number is written
                    *: matches arbitrary text
            return_only_numbering:
                whether to only return the numbering of the matched files
            return_only_file_names:
                whether to only return the name of the matched files

        Returns
        -------
            A sorted list of (numbering, file name) pairs for each matching file/folder in the passed directory
        """

        assert not (return_only_numbering and return_only_file_names), \
            "Can only set one of return_only_numbering and return_only_file_names"

        regex = self._build_numbering_extraction_regex(name_format)
        if Path(self._location).is_dir():
            file_names = [file.name for file in Path(self._location).iterdir()]
        else:
            file_names = []

        file_names_and_numbering = [(int(regex.search(file_name).group(1)), file_name)
                                    for file_name in file_names if regex.match(file_name)]
        file_names_and_numbering = sorted(file_names_and_numbering, key=lambda x: x[0])

        if return_only_numbering:
            return [file_name_and_numbering[0] for file_name_and_numbering in file_names_and_numbering]
        elif return_only_file_names:
            return [file_name_and_numbering[1] for file_name_and_numbering in file_names_and_numbering]
        else:
            return file_names_and_numbering

    def get_file_name_by_numbering(self, name_format: str, numbering: int) -> Optional[str]:
        """
        Obtains the corresponding file name given its numbering.

        Parameters
        ----------
            name_format:
                The assumed format of ordered files. See list_file_numbering()
            numbering:
                the numbering of the desired file

        Returns
        -------
            - The corresponding file name of the given numbering if it could be found
            - None, otherwise
        """

        file_numberings_and_names = self.list_file_numbering(name_format)
        for file_numbering, name in file_numberings_and_names:
            if file_numbering == numbering:
                return name

        # File numbering could not be found
        return None

    def get_numbering_by_file_name(self, name_format: str, file_name: str) -> Optional[int]:
        """
        Extracts the numbering of a file name. The file does not have to exist.

        Parameters
        ----------
            name_format:
                The assumed format of ordered files. See list_file_numbering()
            file_name:
                the name of the desired file for which the numbering should be extracted

        Returns
        -------
            - The corresponding numbering of the file if it could be extracted
            - None, otherwise
        """

        regex = self._build_numbering_extraction_regex(name_format)
        if regex.match(file_name):
            return int(regex.search(file_name).group(1))
        else:
            return None

    @staticmethod
    def substitute(name_format: str, numbering: int, name: Optional[str] = None) -> str:
        """
        Substitutes numbering and optional free text names into the given `name_format`.
        Replaces $ with numbering and potential * with name.

        Parameters
        ----------
            name_format:
                format used as a basis for substitution
            numbering:
                numbering to substitute in
            name:
                optional free text name to substitute in

        Returns
        -------
            The name format with any $ and * wildcards replaced by numbering and name
        """

        wildcard_present = '*' in name_format
        wildcard_optional = '[' in name_format and ']' in name_format \
                            and name_format.index('[') < name_format.index('*') < name_format.index(']')
        name_none = name is None

        assert wildcard_optional or wildcard_present ^ name_none, \
            "If `name` is given, `*` should appear in `name_format` and vice-versa"
        assert name_format.count('*') <= 1, 'Wildcard `*` cannot appear more than once'
        substituted_name = name_format.replace('$', f'{numbering}')

        if wildcard_present and not name_none:
            substituted_name = substituted_name.replace('*', name)

            if wildcard_optional:
                # Just remove square brackets
                substituted_name = substituted_name.replace('[', '').replace(']', '')
        elif wildcard_optional:
            # No name given, but optional wildcard specified. Remove everything between square brackets [...]
            substituted_name = substituted_name[:substituted_name.index('[')] + substituted_name[
                                                                                substituted_name.index(']') + 1:]

        return substituted_name

    def generate_next_name(self,
                           name_format: str,
                           name: Optional[str] = None,
                           create_folder: bool = True) -> str:
        """
        Generates a new run name (and per default creates a folder with that name) that follows the given `name_format`
        and has a numbering that is one larger than the highest present in the folder.
        If this will be the first folder matching the given `name_format` its numbering will be 1.
        The `name_format` assumes exactly one `$` symbol for the location of the numbering in the name. Optionally,
        a single `*` can be specified that will be filled with the given `name`.

        Examples
        --------

            Given Folder structure:

            root
             ├── analysis-batch-norm-100-lambda-10
             ├── analysis-batch-norm-50-lambda-9
             ├── epoch-11.ckpt
             ├── epoch--1.ckpt
             ├── P2P-10
             └── P2P-9

            >>> folder = Folder('root')
            >>> folder.generate_next_name('P2P-$')
            >>> >>> 'P2P-11'

            >>> folder.generate_next_name('epoch-$.ckpt', create_folder=False)
            >>> >>> 'epoch-12.ckpt'

            >>> folder.generate_next_name('analysis-*-$', name='batch-norm-no-lambda')
            >>> >>> 'analysis-batch-norm-no-lambda-11'

        Parameters
        ----------
            name_format:
                Specifies which folders/files in the directory will be considered for the numbering and also formats
                the newly generated run name.
                    $: location of numbering
                    * (optional, max 1): wildcard text. If present, `name` should be set
            name:
                In the case, a `*` wildcard is present in the `name_format` a name has to be specified for the newly
                generated run
            create_folder:
                whether a folder with the newly generated run name should be created automatically

        Returns
        -------
            The name of the new run, ensuring an ascending numbering
        """

        file_numbering = self.list_file_numbering(name_format, return_only_numbering=True)
        if len(file_numbering) == 0:
            new_id = 1
        else:
            max_id = max(file_numbering)
            new_id = max_id + 1 if max_id > 0 else 1  # If only negative IDs are present, use 1

        new_name = self.substitute(name_format, new_id, name=name)

        if create_folder:
            try:
                self.mkdir(new_name)
            except FileExistsError:
                # It can happen that another concurrent run already created that very folder. In this case, just
                # try again
                return self.generate_next_name(name_format, name=name, create_folder=create_folder)

        return new_name

    @staticmethod
    def _build_numbering_extraction_regex(name_format: str) -> re.Pattern:
        assert name_format.count('$') == 1, "The number specifier '$' has to appear in the passed format exactly once"
        assert name_format.count('[') == name_format.count(']'), "square brackets not matching in name format"

        name_format = re.escape(name_format)

        # \[...\] -> (...)?
        name_format = name_format.replace('\\[', '(')
        name_format = name_format.replace('\\]', ')?')

        # $ -> Numbering format
        name_format = name_format.replace(r'\$', r'(-?\d+)')

        # * -> Wildcard format. Make * non-greedy with ?.
        # Otherwise it would match trailing minus signs in '*-$.ckpt' such that $ could never be a negative number
        name_format = name_format.replace(r'\*', r'.*?')

        # Ensure that name_format matches exactly without any leading/trailing leftovers
        name_format = f'^{name_format}$'

        regex = re.compile(name_format)
        return regex

    def __str__(self) -> str:
        return self._location
