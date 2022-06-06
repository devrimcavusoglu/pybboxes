from pybboxes._typing import GenericBboxType
from pybboxes.types.bbox import load_bbox


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
