import numpy as np

from pybboxes.typing import BboxType, GenericBboxType, generic_to_strict


def validate_coco_bbox(bbox: GenericBboxType, strict: bool = True):
    """
    Raises an exception in case of bounding box have incorrect format in COCO style. Does
    nothing otherwise.
    """
    x_tl, y_tl, w, h = bbox
    if w <= 0 or h <= 0:
        raise ValueError("Given width and height must be greater than 0.")
    elif strict and (x_tl < 0 or y_tl < 0):
        raise ValueError("Given top-left point is out of bounds.")


def coco_bbox_to_voc_bbox(bbox: GenericBboxType, **kwargs) -> BboxType:
    """
    Converts COCO style bounding box [x-tl, y-tl, w, h] to
    VOC style bounding box [x-tl, y-tl, x-br, y-br].

    Args:
        bbox: (generic) Bounding box.
        **kwargs: Ignored.

    Returns:
        Bounding box in VOC format.
    """
    validate_coco_bbox(bbox)
    x_tl, y_tl, w, h = bbox
    x_br = x_tl + w
    y_br = y_tl + h
    bbox = np.array([x_tl, y_tl, x_br, y_br])
    return generic_to_strict(bbox, dtype=int)
