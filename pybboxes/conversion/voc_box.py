from typing import Union

import numpy as np

from pybboxes.typing import BboxType, GenericBboxType, generic_to_strict


def validate_voc_bbox(bbox: GenericBboxType, strict: bool = True) -> None:
    """
    Raises an exception in case of bounding box have incorrect format in VOC style. Does
    nothing otherwise.
    """
    x_tl, y_tl, x_br, y_br = bbox
    if x_tl > x_br or y_tl > y_br:
        raise ValueError("Incorrect VOC bbox format. Must be in type [x-tl, y-tl, x-br, y-br].")
    elif x_tl == x_br and y_tl == y_br:
        raise ValueError("Given top-left and bottom-right points must be distinct.")
    elif strict and (x_tl < 0 or y_tl < 0):
        raise ValueError(
            "Top-left axes cannot be negative. To silently skip out of bounds cases " "pass 'strict=True'."
        )


def voc_bbox_to_albumentations_bbox(
    bbox: GenericBboxType, image_width: float, image_height: float, **kwargs
) -> BboxType:
    """
    Converts VOC style bounding box [x-tl, y-tl, x-br, y-br] to
    Albumentations style bounding box [x-tl, y-tl, x-br, y-br].

    Args:
        bbox: Bounding box.
        image_width: (int, float) Width of the image to normalize axes.
        image_height: (int, float) Height of the image to normalize axes.

    Returns:
        Bounding box in COCO format.
    """
    validate_voc_bbox(bbox)
    x_tl, y_tl, x_br, y_br = bbox
    x_tl /= image_width
    y_tl /= image_height
    x_br /= image_width
    y_br /= image_height
    bbox = np.array([x_tl, y_tl, x_br, y_br])
    return generic_to_strict(bbox)


def voc_bbox_to_coco_bbox(bbox: GenericBboxType, **kwargs) -> BboxType:
    """
    Converts VOC style bounding box [x-tl, y-tl, x-br, y-br] to
    COCO style bounding box [x-tl, y-tl, w, h].

    Args:
        bbox: Bounding box.

    Returns:
        Bounding box in COCO format.
    """
    validate_voc_bbox(bbox)
    x_tl, y_tl, x_br, y_br = bbox
    w = x_br - x_tl
    h = y_br - y_tl
    bbox = np.array([x_tl, y_tl, w, h])
    return generic_to_strict(bbox, dtype=int)


def voc_bbox_to_fiftyone_bbox(bbox: GenericBboxType, image_width: float, image_height: float, **kwargs) -> BboxType:
    """
    Converts VOC style bounding box [x-tl, y-tl, x-br, y-br] to
    Fiftyone style (normalized) bounding box [x-tl, y-tl, w, h].

    Args:
        bbox: Bounding box.
        image_width: (int, float) Width of the image to normalize axes.
        image_height: (int, float) Height of the image to normalize axes.

    Returns:
        Bounding box in COCO format.
    """
    validate_voc_bbox(bbox)
    x_tl, y_tl, x_br, y_br = bbox
    w = x_br - x_tl
    h = y_br - y_tl

    x_tl /= image_width
    y_tl /= image_height
    w /= image_width
    h /= image_height
    bbox = np.array([x_tl, y_tl, w, h])
    return generic_to_strict(bbox)


def voc_bbox_to_yolo_bbox(
    bbox: GenericBboxType, image_width: Union[int, float], image_height: Union[int, float], **kwargs
) -> BboxType:
    """
    Converts VOC style bounding box [x-tl, y-tl, x-br, y-br] to
    YOLO style (normalized) bounding box [x-c, y-c, w, h].

    Args:
        bbox: (generic) Bounding box.
        image_width: (int, float) Width of the image to normalize axes.
        image_height: (int, float) Height of the image to normalize axes.

    Returns:
        Bounding box in COCO format.
    """
    validate_voc_bbox(bbox)
    x_tl, y_tl, x_br, y_br = bbox
    w = x_br - x_tl
    h = y_br - y_tl
    x_c = x_tl + w / 2
    y_c = y_tl + h / 2

    x_c /= image_width
    y_c /= image_height
    w /= image_width
    h /= image_height
    bbox = np.array([x_c, y_c, w, h])
    return generic_to_strict(bbox)
