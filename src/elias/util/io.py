"""
The io module contains various utility methods for loading/saving Python objects/dicts in various formats.
Currently, there is support for the most frequent data storing formats:
    - JSON (+gzip)
    - pickled Python objects (+gzip)
"""

import gzip
import json
import pickle
from pathlib import Path
from typing import Union

import numpy as np
import yaml
from PIL import Image

from elias.util.fs import ensure_file_ending, ensure_directory_exists_for_file

PathType = Union[str, Path]


# =========================================================================
# Zipped JSON (.json.gz)
# =========================================================================

def save_zipped_json(obj: dict, path: PathType, suffix: str = 'json.gz'):
    """
    Parses the given Python dict into json, zips it and stores it at the specified location.
    Per default, the path will have a suffix 'json.gz'.

    Parameters
    ----------
        obj: dict
            The JSON (represented as python dict) to be zipped and stored
        path: str
            Where to store the file
        suffix: str, default 'json.gz'
            File name suffix
    """

    path = ensure_file_ending(path, suffix)
    ensure_directory_exists_for_file(path)
    with gzip.open(path, 'wb') as f:
        json.dump(obj, f)


def load_zipped_json(path: PathType, suffix: str = 'json.gz') -> dict:
    """
    Loads the specified compressed JSON file from `path` and extracts it.
    Per default, the path name is assumed to have a suffix 'json.gz'.

    Parameters
    ----------
        path: str
            Path to the zipped JSON to load
        suffix: str, default 'json.gz'
            File name suffix

    Returns
    -------
        The contents of the zipped JSON as a Python dict
    """

    path = ensure_file_ending(path, suffix)
    with gzip.open(path, 'rb') as f:
        return json.load(f)


# =========================================================================
# Zipped pickles (.p.gz)
# =========================================================================

def save_zipped_object(obj: object, path: PathType, suffix: str = 'p.gz'):
    """
    Pickles, zips and stores an arbitrary python object at the specified `path`.
    Per default, the file will have a suffix 'json.gz'.

    Parameters
    ----------
        obj: object
            The Python object to be zipped and stored
        path:
            Where to store the file
        suffix: str, default 'p.gz'
            File name suffix
    """

    path = ensure_file_ending(path, suffix)
    ensure_directory_exists_for_file(path)
    with gzip.open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_zipped_object(path: PathType, suffix: str = 'p.gz') -> object:
    """
    Loads zipped and pickled Python object stored at `path`.
    Per default, the file name is assumed to have a suffix 'p.gz'.

    Parameters
    ----------
        path: str
            Path to the zipped Python object that should be loaded
        suffix: str, default 'p.gz'
            File name suffix

    Returns
    -------
        The un-pickled Python object
    """

    path = ensure_file_ending(path, suffix)
    with gzip.open(path, 'rb') as f:
        return pickle.load(f)


# =========================================================================
# Pickled objects (.p)
# =========================================================================

def save_pickled(obj: object, path: PathType, suffix: str = 'p'):
    """
    Pickles and stores an arbitrary python object at the specified `path`.
    Per default, the file will have a suffix 'p'.

    Parameters
    ----------
        obj: object
            The Python object to be pickled and stored
        path: str
            Where to store the file
        suffix: str, default 'p'
            File name suffix
    """

    path = ensure_file_ending(path, suffix)
    ensure_directory_exists_for_file(path)
    with open(f"{path}", 'wb') as f:
        pickle.dump(obj, f)


def load_pickled(path: PathType, suffix: str = 'p') -> object:
    """
    Loads a pickled Python object stored at the specified `path`.
    Per default, the file name is assumed to have a suffix 'p'.

    Parameters
    ----------
        path: str
            Path to the pickled Python object that should be loaded
        suffix: str, default 'p'
            File name suffix

    Returns
    -------
        The un-pickled Python object
    """

    path = ensure_file_ending(path, suffix)
    with open(path, 'rb') as f:
        return pickle.load(f)


# =========================================================================
# JSON (.json)
# =========================================================================

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def save_json(obj: dict, path: PathType, suffix: str = 'json'):
    """
    Stores the given Python dict in JSON format at `path`.
    Per default, the file will have a suffix 'json'.

    Parameters
    ----------
        obj: dict
            The Python dict to store
        path: str
            Where to store the file
        suffix:
            File name suffix
    """

    path = ensure_file_ending(path, suffix)
    ensure_directory_exists_for_file(path)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=4, cls=NumpyEncoder)


def load_json(path: PathType, suffix: str = 'json') -> dict:
    """
    Loads and parses the given JSON file and returns it as a Python dict.
    Per default, the file name is assumed to have a suffix 'json'.

    Parameters
    ----------
        path: str
            Path to the JSON file
        suffix: str, default 'json'
            File name suffix

    Returns
    -------
        Contents of the JSON file as Python dict
    """

    path = ensure_file_ending(path, suffix)
    with open(path, 'r') as f:
        return json.load(f)


# =========================================================================
# YAML (.yaml)
# =========================================================================

def save_yaml(obj: dict, path: PathType, suffix: str = 'yaml'):
    path = ensure_file_ending(path, suffix)
    ensure_directory_exists_for_file(path)
    with open(path, 'w') as f:
        yaml.dump(obj, f)


def load_yaml(path: PathType, suffix: str = 'yaml') -> dict:
    path_with_suffix = ensure_file_ending(path, suffix)
    if not Path(path_with_suffix).exists():
        if suffix == 'yaml':
            # Try loading .yml instead
            return load_yaml(path, 'yml')
        else:
            raise FileNotFoundError(f"Could not load YAML file {path_with_suffix} (Neither with .yaml suffix). "
                                    f"Is the path correct?")

    with open(path_with_suffix, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)


# =========================================================================
# Images
# =========================================================================

def save_img(img: np.ndarray, path: PathType):
    ensure_directory_exists_for_file(path)

    if img.dtype == np.float32:
        # If float array was passed, have to transform [0.0, 1.0] -> [0, ..., 255)
        assert 0 <= img.min() and img.max() <= 1, \
            "passed float array should have values between 0 and 1 to be interpreted as image"
        img = (img * 255).astype(np.uint8)

    Image.fromarray(img).save(path)


def load_img(path: PathType) -> np.ndarray:
    img = Image.open(path)
    return np.asarray(img)
