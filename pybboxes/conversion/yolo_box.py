import numpy as np

from pybboxes.typing import BboxType, GenericBboxType, generic_to_strict


def validate_yolo_bbox(bbox: GenericBboxType, strict: bool = True):
    """
    Raises an exception in case of bounding box have incorrect format in COCO style. Does
    nothing otherwise.
    """
    x_c, y_c, w, h = bbox
    if (not 0 < w <= 1) or (not 0 < h <= 1):
        raise ValueError("Given width and height must be in range (0,1].")
    elif strict and (not w / 2 <= x_c <= 1 - w / 2 or not h / 2 <= y_c <= 1 - h / 2):
        raise ValueError(
            "Given top-left point is out of bounds. To silently skip out of bounds cases pass 'strict=False'."
        )


def yolo_bbox_to_voc_bbox(bbox: GenericBboxType, image_width: float, image_height: float, **kwargs) -> BboxType:
    """
    Converts YOLO style (normalized) bounding box [x-c, y-c, w, h] to
    VOC style bounding box [x-tl, y-tl, x-br, y-br].

    Args:
        bbox: Bounding box.
        image_width: (int, float) Width of the image to normalize axes.
        image_height: (int, float) Height of the image to normalize axes.

    Returns:
        Bounding box in VOC format.
    """
    validate_yolo_bbox(bbox)
    x_c, y_c, w, h = bbox
    x_tl = x_c - w / 2
    y_tl = y_c - h / 2
    x_tl *= image_width
    y_tl *= image_height
    w *= image_width
    h *= image_height
    x_br = x_tl + w
    y_br = y_tl + h
    bbox = np.array([x_tl, y_tl, x_br, y_br])
    return generic_to_strict(bbox, dtype=int)
