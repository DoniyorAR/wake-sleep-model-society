"""JSON helpers."""
import json
import os


def read_json(path):
    with open(path) as fh:
        return json.load(fh)


def write_json(obj, path, **kwargs):
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, **kwargs)
