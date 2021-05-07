import gzip
import json
import pickle

from elias.fs import ensure_file_ending, create_directories


def save_zipped_json(obj: object, file: str):
    file = ensure_file_ending(file, "json.gz")
    create_directories(file)
    with gzip.open(file, 'wb') as f:
        json.dump(obj, f)


def load_zipped_json(file: str) -> dict:
    file = ensure_file_ending(file, "json.gz")
    with gzip.open(file, 'rb') as f:
        return json.load(f)


def save_zipped_object(obj, file):
    file = ensure_file_ending(file, "p.gz")
    create_directories(file)
    with gzip.open(file, 'wb') as f:
        pickle.dump(obj, f)


def load_zipped_object(file) -> object:
    file = ensure_file_ending(file, "p.gz")
    with gzip.open(file, 'rb') as f:
        return pickle.load(f)


def save_pickled(obj, file):
    file = ensure_file_ending(file, "p")
    create_directories(file)
    with open(f"{file}", 'wb') as f:
        pickle.dump(obj, f)


def load_pickled(file) -> object:
    file = ensure_file_ending(file, "p")
    with open(file, 'rb') as f:
        return pickle.load(f)


def save_json(obj: dict, file):
    file = ensure_file_ending(file, "json")
    create_directories(file)
    with open(file, 'w') as f:
        json.dump(obj, f, indent=4)


def load_json(file) -> dict:
    file = ensure_file_ending(file, "json")
    with open(file, 'r') as f:
        return json.load(f)
