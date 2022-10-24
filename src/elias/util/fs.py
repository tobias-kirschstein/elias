import os
from pathlib import Path


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


def ensure_directory_exists_for_file(path: str):
    """
    Ensures that the folder to the specified path exists and creates a nested folder structure if necessary.
    Be careful to use trailing '/' if `path` directly constitutes the target folder (and not an arbitrary file
    within that folder).

    Parameters
    ----------
        path: path to the file or folder for which an underlying directory structure will be ensured
    """

    Path(os.path.dirname(str(path))).mkdir(parents=True, exist_ok=True)


def ensure_directory_exists(path: str):
    """
    Ensures that the specified folder exists and creates a nested folder structure if necessary.

    Parameters
    ----------
        path: path to the folder which should exist
    """

    Path(str(path)).mkdir(parents=True, exist_ok=True)


def clear_directory(path: str):
    """
    Clears all files from the specified directory, but keeps the directory itself.
    If the directory does not exist, nothing happens.
    """

    path = Path(str(path))
    if path.exists():
        for f in Path(path).iterdir():
            f.unlink()
