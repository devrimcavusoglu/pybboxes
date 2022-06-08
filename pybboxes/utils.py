import importlib.util
from pathlib import Path
from typing import Union


def import_module(module_name: str, filepath: Union[str, Path]):
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
