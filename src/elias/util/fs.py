import os
import re
from pathlib import Path
from shutil import copy2
from typing import Callable, Union

import PIL
from PIL import Image

from elias.util.range import IndexRangeBundle


def ensure_file_ending(file: str, ending: str) -> str:
    """
    Ensures that the given file path has the specified ending.

    Parameters
    ----------
        file: path to file
        ending: extension that the file should have

    Returns
    -------
        the file path with the specified extension if it did not already have that ending
    """

    return file if file.endswith(ending) else f"{file}.{ending}"


def ensure_directory_exists_for_file(path: Union[str, Path]):
    """
    Ensures that the folder to the specified path exists and creates a nested folder structure if necessary.
    Be careful to use trailing '/' if `path` directly constitutes the target folder (and not an arbitrary file
    within that folder).

    Parameters
    ----------
        path: path to the file or folder for which an underlying directory structure will be ensured
    """

    Path(os.path.dirname(str(path))).mkdir(parents=True, exist_ok=True)


def ensure_directory_exists(path: Union[str, Path]):
    """
    Ensures that the specified folder exists and creates a nested folder structure if necessary.

    Parameters
    ----------
        path: path to the folder which should exist
    """

    Path(str(path)).mkdir(parents=True, exist_ok=True)


def clear_directory(path: Union[str, Path]):
    """
    Clears all files from the specified directory, but keeps the directory itself.
    If the directory does not exist, nothing happens.
    """

    path = Path(str(path))
    if path.exists():
        for f in Path(path).iterdir():
            f.unlink()


# ==========================================================
# Copy utils
# ==========================================================

def _get_index_range_lambda(range_specifier: str) -> Callable[[int], bool]:
    if range_specifier == "":
        return lambda number: True
    else:
        index_range_bundle = IndexRangeBundle.from_description(range_specifier)
        return lambda number: number in index_range_bundle
        # range_specifier = range_specifier.replace("\\", "")  # Get rid of escape characters \\
        # index_ranges = [IndexRange.from_description(description) for description in
        #                 range_specifier.split(',')]
        # return lambda number: any([number in index_range for index_range in index_ranges])


def copy_transfer_fn(source_path: str, target_path: str):
    print(source_path, " -> ", target_path)
    ensure_directory_exists_for_file(target_path)
    copy2(source_path, target_path)


def copy_and_downsample_transfer_fn(source_path: str,
                                    target_path: str,
                                    downsample_factor: float):
    print(source_path, " -> ", target_path)
    ensure_directory_exists_for_file(target_path)

    image = Image.open(source_path)
    downsampled_size = (int(image.size[0] / downsample_factor),
                        int(image.size[1] / downsample_factor))
    image = image.resize(downsampled_size,
                         resample=PIL.Image.Resampling.LANCZOS)  # TODO: Is this the best resampling?
    image.save(target_path)


def smart_copy(
        source_files_format: str,
        target_files_format: str,
        transfer_fn: Callable[[str, str], None] = copy_transfer_fn):
    """
    Copies all the files that match the specified `source_files_format` and copies + renames them according to
    the wildcards used in `target_files_format`.

    Parameters
    ----------
        source_files_format:
            wildcarded path to source files, .e.g., C:/CaptureData/005/{s:}/cam_{c:}/image_{t:}.png
        target_files_format:
            wildcarded path to destination where files should be copied.
            All wildcards used in `source_files_format` must be reused.
            E.g., H:/CaptureData/005/sequence_{s:}_image_{t:}_cam_{c:}.png
        transfer_fn:
            A function that is executed for each pair of source/target paths.
            The default transfer_fn simply copies the source to the target file.
            Other transfer functions such as move instead of copy or copy + downsample are possible as well

    """

    # Traverse source file path up until the path does not contain any wildcards anymore.
    # From here, we can list all sub files and will find all relevant matches
    root_folder = Path(source_files_format)
    contains_wildcard_pattern = re.compile(r"{.*?:.*?}")
    while contains_wildcard_pattern.search(str(root_folder)):
        root_folder = root_folder.parent

    # Format now only contains parts with wildcards
    source_files_format = os.path.relpath(source_files_format, root_folder)

    print("Detected Root Folder:", root_folder)
    print("Copy files that match:", source_files_format)

    # Translate specified f-string into a regex
    # {var:} -> ((?P<var>.*)
    # This is necessary, as we cannot match f-strings against file names, but with regexes we can
    source_files_format = re.escape(source_files_format)

    decimal_wildcard_pattern = re.compile(r"\\{([^{}]*?):([^{}]*?)d\\}")
    generic_wildcard_pattern = re.compile(r"\\{([^{}]]*?):[^{}]*?\\}")
    decimal_wildcards = {wildcard: range_specifier
                         for wildcard, range_specifier
                         in decimal_wildcard_pattern.findall(source_files_format)}
    for wildcard, range_specifier in decimal_wildcards.items():
        decimal_wildcards[wildcard] = _get_index_range_lambda(range_specifier)

    source_files_regex = re.sub(decimal_wildcard_pattern, r"(?P<\g<1>>\\d*?)", source_files_format)
    source_files_regex = re.sub(generic_wildcard_pattern, r"(?P<\g<1>>.*?)", source_files_regex)
    print("Regex: ", source_files_regex)
    source_files_regex = re.compile(source_files_regex)

    for path in root_folder.rglob("*"):
        relevant_path = os.path.relpath(path, root_folder)
        match = re.match(source_files_regex, relevant_path)
        if match:
            match_arguments = match.groupdict()
            skip = False
            for key, value in match_arguments.items():
                if key in decimal_wildcards:
                    value = int(value)
                    include_value = decimal_wildcards[key]
                    if not include_value(value):
                        skip = True

                    match_arguments[key] = int(value)

            if not skip:
                target_path = target_files_format.format(**match_arguments)
                transfer_fn(path, target_path)
