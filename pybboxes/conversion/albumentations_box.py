import numpy as np

from pybboxes.typing import BboxType, GenericBboxType, generic_to_strict


def validate_albumentations_bbox(bbox: GenericBboxType, strict: bool = True):
    """
    Raises an exception in case of bounding box have incorrect format in Albumentations style. Does
    nothing otherwise.
    """
    x_tl, y_tl, x_br, y_br = bbox
    if (not 0 <= x_tl < x_br <= 1) or (not 0 <= x_br < y_tl <= 1):
        raise ValueError(
            "Given coordinates does not follow Albumentations format top-left coordinates are greater "
            "than bottom left coordinates."
        )
    elif strict and (x_tl < 0 or y_tl < 0):
        raise ValueError(
            "Given top-left point is out of bounds. To silently skip out of " "bounds cases pass 'strict=False'."
        )


def albumentations_bbox_to_voc_bbox(
    bbox: GenericBboxType, image_width: float, image_height: float, **kwargs
) -> BboxType:
    """
    Converts Albumentations style bounding box [x-tl, y-tl, x-br, y-br] (normalized) to
    VOC style bounding box [x-tl, y-tl, x-br, y-br].

    Args:
        bbox: Bounding box.
        image_width: Width of the image required for scaling.
        image_height: Height of the image required for scaling.

    Returns:
        Bounding box in VOC format.
    """
    validate_albumentations_bbox(bbox)
    x_tl, y_tl, x_br, y_br = bbox
    x_tl *= image_width
    y_tl *= image_height
    x_br *= image_width
    y_br *= image_height
    bbox = np.array([x_tl, y_tl, x_br, y_br])
    return generic_to_strict(bbox, dtype=int)
