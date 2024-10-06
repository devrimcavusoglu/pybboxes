import json
import os
import struct
from typing import Dict, List, Optional, Union

import yaml


def get_image_size(file_path: str):
    """
    Return (width, height) for a given img file content - no external
    dependencies except the os and struct modules from Python core
    """
    with open(file_path, "rb") as fhandle:
        head = fhandle.read(24)
        if len(head) != 24:
            return None
        if head.startswith(b"\x89PNG\r\n\x1a\n"):
            check = struct.unpack(">i", head[16:20])[0]
            if check != 0x0D0A1A0A:
                return None
            width, height = struct.unpack(">ii", head[16:24])
        elif head[:2] == b"\xff\xd8":
            try:
                fhandle.seek(0)  # Read 0xff next
                size = 2
                ftype = 0
                while not 0xC0 <= ftype <= 0xCF:
                    fhandle.seek(size, 1)
                    byte = fhandle.read(1)
                    while ord(byte) == 0xFF:
                        byte = fhandle.read(1)
                    ftype = ord(byte)
                    size = struct.unpack(">H", fhandle.read(2))[0] - 2
                # We are at a SOFn block
                fhandle.seek(1, 1)  # Skip `precision' byte.
                height, width = struct.unpack(">HH", fhandle.read(4))
            except Exception:
                return None
        else:
            return None
        return width, height


class IndentfulDumper(yaml.Dumper):
    def increase_indent(self, flow=False, indentless=False):
        return super(IndentfulDumper, self).increase_indent(flow, False)


def assure_overridable(f):
    """
    Wrapper allowing easy use of overwrite-safe functionality. All of the write-helpers
    use this wrapper. In case of a conflict, it raises an exception.
    """

    def wrapper(obj, fp, **kwargs):
        overwrite = kwargs.get("overwrite", True)
        if os.path.exists(fp) and not overwrite:
            raise ValueError(f"Path {fp} already exists. To overwrite, use `overwrite=True`.")
        return f(obj, fp, **kwargs)

    return wrapper


def read_json(fp: str, **kwargs) -> Union[Dict, List]:
    """
    Reads a JSON file given path.

    Args:
        fp: (str) File path.

    Return:
        Dictionary or List of dictionaries depending on the content.
    """
    with open(fp, "r") as fd_in:
        data = json.load(fd_in, **kwargs)
    return data


def read_yaml(fp: str) -> Union[Dict, List]:
    """
    Reads a YAML file given path.

    Args:
        fp: (str) File path.

    Return:
        Generic Python object.
    """
    with open(fp, "r") as fd_in:
        data = yaml.safe_load(fd_in)
    return data


@assure_overridable
def write_json(obj: Union[Dict, List], fp: str, encoding: Optional[str] = None, **kwargs) -> None:
    """
    Writes a Python dictionary or list object to the given path in JSON format.

    Args:
        obj: (dict, list) Python dictionary or list object.
        fp: (str) Path of the output file.
        encoding: (Optional(str)) Encoding for writing.
    """
    with open(fp, "w", encoding=encoding) as fd_out:
        json.dump(obj, fd_out, **kwargs)


@assure_overridable
def write_yaml(obj: Dict, fp: str, indent_blocks: bool = True, **kwargs) -> None:
    """
    Writes a Python dictionary to the given path in YAML format.

    Args:
        obj: (any) Serializable Python object.
        fp: (str) Path of the output file.
        indent_blocks: (bool) Whether dump with indents.
    """
    with open(fp, "w") as fd_out:
        if indent_blocks:
            d = yaml.dump(obj, Dumper=IndentfulDumper, **kwargs)
            fd_out.write(d)
        else:
            yaml.safe_dump(obj, fd_out, **kwargs)
