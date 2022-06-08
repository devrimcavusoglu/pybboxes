from typing import Tuple, Union

from pybboxes.types.base import BaseBoundingBox
from pybboxes.types.bbox import load_bbox
from pybboxes.typing import BboxType, GenericBboxType


def convert_bbox(
    bbox: GenericBboxType,
    from_type: str = None,
    to_type: str = None,
    image_size: Tuple[int, int] = None,
    return_values: bool = True,
    **kwargs,
) -> Union[BboxType, BaseBoundingBox]:
    """
    Converts given bbox with given `from_type` to given `to_type`. It uses VOC format
    as an intermediate format.

    Args:
        bbox: (generic) Bounding box.
        from_type: (str) Type/Format of the given bounding box.
        to_type: (str) Type/Format of the resulting bounding box.
        image_size: (tuple(int,int)) Image size as (w, h) tuple, it is required if the one side of the
            types requires scaling.
        return_values: (bool) Whether to return values as a Tuple, or BoundingBox object.
            True by default for compatibility purposes.

    Return:
        Bounding box in type `to_type`.
    """
    if not isinstance(bbox, BaseBoundingBox):
        if not from_type:
            raise ValueError("if `bbox` is not a BoundingBox object, `from_type` is required.")
        bbox = load_bbox(name=from_type, values=bbox, image_size=image_size, **kwargs)
    source_to_target = getattr(bbox, f"to_{to_type}")
    target_bbox = source_to_target()
    if return_values:
        return target_bbox.values
    return target_bbox


def compute_intersection(bbox1: GenericBboxType, bbox2: GenericBboxType, bbox_type: str = "coco", **kwargs):
    """
    Computes intersection area between given bounding boxes.

    Args:
        bbox1: Bounding box 1.
        bbox2: Bounding box 2.
        bbox_type: Format of the bounding boxes. It's 'coco' [x-tl, y-tl, w, h] by default.

    Returns:
        Intersection area if bounding boxes intersect, 0 otherwise.
    """
    bbox1 = load_bbox(name=bbox_type, values=bbox1, **kwargs)
    bbox2 = load_bbox(name=bbox_type, values=bbox2, **kwargs)
    return bbox1 * bbox2


def compute_area(bbox: GenericBboxType, bbox_type: str = "coco", **kwargs):
    """
    Computes the area of given bounding box.
    """
    return compute_intersection(bbox, bbox, bbox_type, **kwargs)


def compute_union(bbox1: GenericBboxType, bbox2: GenericBboxType, bbox_type: str = "coco", **kwargs):
    """
    Computes union area of given boxes.

    Args:
        bbox1: Bounding box 1.
        bbox2: Bounding box 2.
        bbox_type: Format of the bounding boxes. It's 'coco' [x-tl, y-tl, w, h] by default.

    Returns:
        Union area.
    """
    intersection = compute_intersection(bbox1, bbox2, bbox_type=bbox_type, **kwargs)
    area1 = compute_area(bbox1, bbox_type=bbox_type, **kwargs)
    area2 = compute_area(bbox2, bbox_type=bbox_type, **kwargs)
    return area1 + area2 - intersection


def compute_iou(bbox1: GenericBboxType, bbox2: GenericBboxType, bbox_type: str = "coco", **kwargs):
    """
    Computes Intersection over Union (IoU) (special form of Jaccard Index) metric.

    Args:
        bbox1: Bounding box 1.
        bbox2: Bounding box 2.
        bbox_type: Format of the bounding boxes. It's 'coco' [x-tl, y-tl, w, h] by default.

    Returns:
        Intersection over Union ratio.
    """
    return compute_intersection(bbox1, bbox2, bbox_type, **kwargs) / compute_union(bbox1, bbox2, bbox_type, **kwargs)
