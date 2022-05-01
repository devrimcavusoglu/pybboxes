import numpy as np

from pybboxes.typing import BboxType, GenericBboxType, generic_to_strict


def validate_fiftyone_bbox(bbox: GenericBboxType, strict: bool = True):
    """
    Raises an exception in case of bounding box have incorrect format in FiftyOne style. Does
    nothing otherwise.
    """
    x_tl, y_tl, w, h = bbox
    if (not 0 < w <= 1) or (not 0 < h <= 1):
        raise ValueError("Given width and height must be in range (0,1].")
    elif strict and (x_tl < 0 or y_tl < 0):
        raise ValueError(
            "Given top-left point is out of bounds. To silently skip out of " "bounds cases pass 'strict=False'."
        )
    elif (not 0 <= x_tl <= 1) or (not 0 <= y_tl <= 1):
        raise ValueError(
            "Incorrect FiftyOne style bounding-box, where (x-topleft, y-topleft) " "are not in the range [0,1]."
        )


def fiftyone_bbox_to_voc_bbox(bbox: GenericBboxType, image_width: float, image_height: float, **kwargs) -> BboxType:
    """
    Converts FiftyOne style bounding box [x-tl, y-tl, w, h] to
    VOC style bounding box [x-tl, y-tl, x-br, y-br].

    Args:
        bbox: Bounding box.
        image_width: Width of the image required for scaling.
        image_height: Height of the image required for scaling.

    Returns:
        Bounding box in VOC format.
    """
    validate_fiftyone_bbox(bbox)
    x_tl, y_tl, w, h = bbox
    x_tl *= image_width
    y_tl *= image_height
    w *= image_width
    h *= image_height
    x_br = x_tl + w
    y_br = y_tl + h
    bbox = np.array([x_tl, y_tl, x_br, y_br])
    return generic_to_strict(bbox, dtype=int)
