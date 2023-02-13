import importlib.util
import inspect
import os.path
from pathlib import Path
from typing import Union


def import_module(module_name: str, filepath: Union[str, Path]):
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def find_stack_level() -> int:
    """
    Taken and adapted from pandas exception utility module.
    ref:
    https://github.com/pandas-dev/pandas/blob/22cb3793b47ed5b1f98156b58e0bfc109acebdc9/pandas/util/_exceptions.py#L27
    """

    import pybboxes as pbx

    pkg_dir = os.path.dirname(pbx.__file__)
    test_dir = os.path.join(pkg_dir, "tests")

    # https://stackoverflow.com/questions/17407119/python-inspect-stack-is-slow
    frame = inspect.currentframe()
    n = 0
    while frame:
        fname = inspect.getfile(frame)
        if fname.startswith(pkg_dir) and not fname.startswith(test_dir):
            frame = frame.f_back
            n += 1
        else:
            break
    return n
