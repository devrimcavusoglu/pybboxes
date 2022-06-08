from importlib import import_module
from typing import Tuple, Union

import humps

from pybboxes.types.base import BaseBoundingBox


def load_bbox(
    name: str, values, image_size: Tuple[int, int] = None, return_values: bool = False, from_voc: bool = False, **kwargs
) -> BaseBoundingBox:
    module_name = f"{name}_bounding_box"
    module_path = f"pybboxes.types.{module_name}"
    klass_name = humps.pascalize(module_name)
    module = import_module(module_path)
    klass = getattr(module, klass_name)
    if from_voc:
        # Used to convert from Generic (VOC) style
        bbox = klass.from_voc(*values, image_size=image_size, **kwargs)
    else:
        bbox = klass(*values, image_size=image_size, **kwargs)
    if return_values:
        return bbox.values
    return bbox


class BoundingBox(BaseBoundingBox):
    def __init__(
        self,
        x_tl: int,
        y_tl: int,
        x_br: int,
        y_br: int,
        image_size: Tuple[int, int] = None,
        strict: bool = True,
    ):
        super(BoundingBox, self).__init__(x_tl, y_tl, x_br, y_br, image_size=image_size, strict=strict)

    def _correct_value_types(self, x_tl: int, y_tl: int, x_br: int, y_br: int) -> Tuple:
        return round(x_tl), round(y_tl), round(x_br), round(y_br)

    def _validate_values(self, x_tl: int, y_tl: int, x_br: int, y_br: int):
        image_width, image_height = self.image_size
        if x_tl > x_br or y_tl > y_br:
            raise ValueError("Incorrect BoundingBox format. Must be in type [x-tl, y-tl, x-br, y-br].")
        elif (x_tl, y_tl) == (x_br, y_br):
            raise ValueError("Given top-left and bottom-right points must be distinct.")
        elif (
            self.strict
            and (x_tl < 0 or y_tl < 0)
            or (image_height is not None and y_br > image_height)
            or (image_width is not None and x_br > image_width)
        ):
            raise ValueError(
                "Top-left axes cannot be negative. To silently skip out of bounds cases pass 'strict=False'."
            )

    def _to_bbox_type(self, name: str, return_values: bool) -> BaseBoundingBox:
        return load_bbox(
            name, values=self.values, image_size=self.image_size, return_values=return_values, from_voc=True
        )

    def to_albumentations(self, return_values: bool = False) -> Union[Tuple[int, int, int, int], "BaseBoundingBox"]:
        return self._to_bbox_type("albumentations", return_values)

    def to_coco(self, return_values: bool = False) -> Union[Tuple[int, int, int, int], "BaseBoundingBox"]:
        return self._to_bbox_type("coco", return_values)

    def to_fiftyone(self, return_values: bool = False) -> Union[Tuple[int, int, int, int], "BaseBoundingBox"]:
        return self._to_bbox_type("fiftyone", return_values)

    def to_voc(self, return_values: bool = False) -> Union[Tuple[int, int, int, int], "BaseBoundingBox"]:
        if return_values:
            return self.values
        return self

    def to_yolo(self, return_values: bool = False) -> Union[Tuple[int, int, int, int], "BaseBoundingBox"]:
        return self._to_bbox_type("yolo", return_values)

    @classmethod
    def from_voc(
        cls,
        x_tl: int,
        y_tl: int,
        x_br: int,
        y_br: int,
        image_size: Tuple[int, int] = None,
        strict: bool = True,
    ) -> "BoundingBox":
        return cls(x_tl, y_tl, x_br, y_br, image_size=image_size, strict=strict)

    @classmethod
    def from_albumentations(
        cls, x_tl: float, y_tl: float, x_br: float, y_br: float, image_size: Tuple[int, int] = None, strict: bool = True
    ):
        return load_bbox("albumentations", values=(x_tl, y_tl, x_br, y_br), image_size=image_size, strict=strict)

    @classmethod
    def from_coco(cls, x_tl: int, y_tl: int, w: int, h: int, image_size: Tuple[int, int] = None, strict: bool = True):
        return load_bbox("coco", values=(x_tl, y_tl, w, h), image_size=image_size, strict=strict)

    @classmethod
    def from_fiftyone(
        cls, x_tl: float, y_tl: float, w: float, h: float, image_size: Tuple[int, int] = None, strict: bool = True
    ):
        return load_bbox("fiftyone", values=(x_tl, y_tl, w, h), image_size=image_size, strict=strict)

    @classmethod
    def from_yolo(
        cls, x_c: float, y_c: float, w: float, h: float, image_size: Tuple[int, int] = None, strict: bool = True
    ):
        return load_bbox("yolo", values=(x_c, y_c, w, h), image_size=image_size, strict=strict)
