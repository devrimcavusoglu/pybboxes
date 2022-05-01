import importlib

from pybboxes.typing import BboxType, GenericBboxType


def convert_bbox(bbox: GenericBboxType, from_type: str, to_type: str, **kwargs) -> BboxType:
    """
    Converts given bbox with given `from_type` to given `to_type`. It uses VOC format
    as an intermediate format.

    Args:
        bbox: (generic) Bounding box.
        from_type: (str) Type/Format of the given bounding box.
        to_type: (str) Type/Format of the resulting bounding box.

    Return:
        Bounding box in type `to_type`.
    """
    if from_type == to_type:
        return bbox

    modules_root = "pybboxes.conversion"
    if from_type != "voc":
        source_module = importlib.import_module(f"{modules_root}.{from_type}_box")
        source_to_voc = getattr(source_module, f"{from_type}_bbox_to_voc_bbox")
    else:
        source_to_voc = None

    if source_to_voc is not None:
        bbox = source_to_voc(bbox, **kwargs)

    if to_type != "voc":
        target_module = importlib.import_module(f"{modules_root}.voc_box")
        voc_to_target = getattr(target_module, f"voc_bbox_to_{to_type}_bbox")
        bbox = voc_to_target(bbox, **kwargs)

    return bbox
