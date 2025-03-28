"""
The io module contains various utility methods for loading/saving Python objects/dicts in various formats.
Currently, there is support for the most frequent data storing formats:
    - JSON (+gzip)
    - pickled Python objects (+gzip)
"""

import gzip
import io
import json
import os
import pickle
import urllib
from io import BytesIO
from pathlib import Path
from typing import Union, Tuple, Optional
from urllib.error import HTTPError, URLError

from matplotlib import pyplot as plt

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import PIL.Image
import pillow_avif  # IMPORTANT: This import must be here, otherwise .avif files won't be loaded with "PIL.UnidentifiedImageError: cannot identify image file"
import cv2
import imageio
import numpy as np
import requests
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

def save_img(img: np.ndarray, path: PathType, quality: Optional[int] = None):
    """
    Save the given numpy array as an 8-bit image. The image type is determined by the file extension.
    `image` should be a np.uint8 array with shape [H, W, (C)] with C in {1, 3, 4} and values in range [0, 255]
    If `image` has type np.float32, it is assumed that values are in range [0, 1]. The float image will be automatically converted to uint8 type.

    Parameters
    ----------
        img: image as a [H, W, (C)] numpy array with values in [0, 255] or [0, 1].
        path: where to store the image. Any necessarily folders will be created automatically.
        quality: quality for JPEG compression. Default takes the default compression level from pillow (75)
    """

    ensure_directory_exists_for_file(path)

    if Path(path).suffix == '.exr':
        imageio.imwrite(path, img)
    else:
        if img.dtype == np.float32:
            # TODO: Do we really want to quantize without asking?
            # If float array was passed, have to transform [0.0, 1.0] -> [0, ..., 255)
            assert 0 <= img.min() and img.max() <= 1, \
                "passed float array should have values between 0 and 1 to be interpreted as image"
            img = (img * 255).astype(np.uint8)

        if len(img.shape) == 3 and img.shape[2] == 1:
            # Assume this should be stored as a single-channel grayscale image
            img = img[:, :, 0]

        if quality is None:
            Image.fromarray(img).save(path)
        else:
            Image.fromarray(img).save(path, quality=quality)


def load_img(path: PathType) -> np.ndarray:
    if Path(path).suffix == '.exr':
        img = imageio.imread_v2(path)
    else:
        img = Image.open(path)

    return np.asarray(img)


def download_img(url: str) -> np.ndarray:
    response = requests.get(url)
    pil_img = Image.open(BytesIO(response.content))
    img = np.asarray(pil_img)

    return img


InterpolationType = Literal['nearest', 'bilinear', 'bicubic', 'lanczos']


def resize_img(img: np.ndarray,
               scale: Union[float, Tuple[float, float]],
               interpolation: InterpolationType = 'bilinear',
               use_opencv: bool = False) -> np.ndarray:
    try:
        iter(scale)
    except TypeError:
        scale_x = scale
        scale_y = scale
    else:
        scale_x, scale_y = scale

    if use_opencv:
        if interpolation == 'nearest':
            interpolation = cv2.INTER_NEAREST
        elif interpolation == 'bilinear':
            interpolation = cv2.INTER_LINEAR
        elif interpolation == 'bicubic':
            interpolation = cv2.INTER_CUBIC
        elif interpolation == 'lanczos':
            interpolation = cv2.INTER_LANCZOS4
        else:
            raise ValueError(f"Invalid interpolation type: {interpolation}")

        # NB: Python's round() does round-to-even! Regular round can be implemented by int(x + 0.5)
        img = cv2.resize(img, (int(img.shape[1] * scale_x + 0.5), int(img.shape[0] * scale_y + 0.5)), interpolation=interpolation)
    else:
        if interpolation == 'nearest':
            interpolation = PIL.Image.Resampling.NEAREST
        elif interpolation == 'bilinear':
            interpolation = PIL.Image.Resampling.BILINEAR
        elif interpolation == 'bicubic':
            interpolation = PIL.Image.Resampling.BICUBIC
        elif interpolation == 'lanczos':
            interpolation = PIL.Image.Resampling.LANCZOS
        else:
            raise ValueError(f"Invalid interpolation type: {interpolation}")

        img_pil = Image.fromarray(img)
        # NB: Python's round() does round-to-even! Regular round can be implemented by int(x + 0.5)
        img_pil = img_pil.resize((int(img.shape[1] * scale_x + 0.5), int(img.shape[0] * scale_y + 0.5)), resample=interpolation)
        img = np.array(img_pil)

    return img


def fig2img(fig: Optional[plt.Figure] = None, transparent: bool = False) -> np.ndarray:
    """
    Converts a Matplotlib figure to a numpy image.
    If no figure is given, the current open matplot figure will be used instead.
    """

    buf = io.BytesIO()
    if fig is None:
        fig = plt.gcf()

    fig.savefig(buf, bbox_inches='tight', transparent=transparent)
    buf.seek(0)
    img = np.asarray(Image.open(buf))

    if transparent:
        return img[..., :4]
    else:
        return img[..., :3]


def download_file(url: str, target_path: str) -> None:
    ensure_directory_exists_for_file(target_path)

    if Path(target_path).exists():
        response = requests.head(url)
        download_size = int(response.headers['content-length'])
        local_file_size = os.path.getsize(target_path)

        if download_size == local_file_size:
            print(f"{target_path} already exists, skipping")
            return
        else:
            print(f"{target_path} seems to be incomplete. Re-downloading...")

    print(f"Downloading file from {url} to {target_path}")

    try:
        urllib.request.urlretrieve(url, target_path)
    except HTTPError as e:
        print(f"HTTP error occurred reaching {url}: {e}")
    except URLError as e:
        print(f"URL error occurred reaching {url}: {e}")
