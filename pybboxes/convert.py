from typing import Tuple, Union

from pybboxes._typing import BboxType, GenericBboxType
from pybboxes.types.base import BaseBoundingBox
from pybboxes.types.bbox import load_bbox


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
