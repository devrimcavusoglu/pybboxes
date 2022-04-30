import numpy as np

from obsstools.cv.bbox.coco_box import coco_bbox_to_voc_bbox
from obsstools.cv.bbox.convert import convert_bbox
from obsstools.cv.bbox.typing import GenericBboxType
from obsstools.cv.bbox.voc_box import validate_voc_bbox


def compute_intersection(bbox1: GenericBboxType, bbox2: GenericBboxType, bbox_type: str = "coco"):
    """
    Computes intersection area between given bounding boxes.

    Args:
        bbox1: Bounding box 1.
        bbox2: Bounding box 2.
        bbox_type: Format of the bounding boxes. It's 'coco' [x-tl, y-tl, w, h] by default.

    Returns:
        Intersection area if bounding boxes intersect, 0 otherwise.
    """
    if bbox_type in ["coco", "fiftyone"]:
        # COCO and Fiftyone are in the same format of [x-tl,y-tl,w,h]
        # except Fiftyone format is in normalized form in range (0,1).
        # We would need im_width & im_height to convert additionally,
        # to surpass this problem we directly apply conversion from
        # COCO to VOC as this does not affect the area computation.
        bbox1 = coco_bbox_to_voc_bbox(bbox1)
        bbox2 = coco_bbox_to_voc_bbox(bbox2)
    else:
        bbox1 = convert_bbox(bbox1, from_type=bbox_type, to_type="voc")
        bbox2 = convert_bbox(bbox2, from_type=bbox_type, to_type="voc")
    validate_voc_bbox(bbox1)
    validate_voc_bbox(bbox2)
    x_tl, y_tl = np.maximum(bbox1[:2], bbox2[:2])
    x_br, y_br = np.minimum(bbox1[2:], bbox2[2:])
    if x_tl >= x_br or y_tl >= y_br:
        return 0
    width = x_br - x_tl
    height = y_br - y_tl
    return width * height


def compute_area(bbox: GenericBboxType, bbox_type: str = "coco"):
    """
    Computes the area of given bounding box.
    """
    return compute_intersection(bbox, bbox, bbox_type)


def compute_union(bbox1: GenericBboxType, bbox2: GenericBboxType, bbox_type: str = "coco"):
    """
    Computes union area of given boxes.

    Args:
        bbox1: Bounding box 1.
        bbox2: Bounding box 2.
        bbox_type: Format of the bounding boxes. It's 'coco' [x-tl, y-tl, w, h] by default.

    Returns:
        Union area.
    """
    intersection = compute_intersection(bbox1, bbox2, bbox_type=bbox_type)
    area1 = compute_area(bbox1, bbox_type=bbox_type)
    area2 = compute_area(bbox2, bbox_type=bbox_type)
    return area1 + area2 - intersection


def compute_iou(bbox1: GenericBboxType, bbox2: GenericBboxType, bbox_type: str = "coco"):
    """
    Computes Intersection over Union (IoU) (special form of Jaccard Index) metric.

    Args:
        bbox1: Bounding box 1.
        bbox2: Bounding box 2.
        bbox_type: Format of the bounding boxes. It's 'coco' [x-tl, y-tl, w, h] by default.

    Returns:
        Intersection over Union ratio.
    """
    return compute_intersection(bbox1, bbox2, bbox_type) / compute_union(bbox1, bbox2, bbox_type)
