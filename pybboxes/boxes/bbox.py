from importlib import import_module
from typing import Tuple, Union

from numpy import sqrt

from pybboxes.boxes.base import BaseBoundingBox


def load_bbox(
    name: str, values, image_size: Tuple[int, int] = None, return_values: bool = False, from_voc: bool = False, **kwargs
) -> BaseBoundingBox:
    def pascalize(snake_string: str) -> str:
        return snake_string.title().replace("_", "")

    module_name = f"{name}_bounding_box"
    module_path = f"pybboxes.boxes.{module_name}"
    klass_name = pascalize(module_name)
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
        strict: bool = False,
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
            not 0 <= x_tl < x_br
            or not 0 <= y_tl < y_br
            or (image_width is not None and x_br > image_width)
            or (image_height is not None and y_br > image_height)
        ):
            if self.strict:
                raise ValueError(
                    "Given bounding box values is out of bounds. "
                    "To silently skip out of bounds cases pass 'strict=False'."
                )
            self._is_oob = True
        elif not self.is_image_size_null():
            self._is_oob = False

    def clamp(self) -> "BoundingBox":
        if self.is_image_size_null() or not self.is_oob:
            return self
        x_tl, y_tl, x_br, y_br = self.raw_values
        width, height = self.image_size
        x_tl = max(x_tl, 0)
        y_tl = max(y_tl, 0)
        x_br = min(x_br, width)
        y_br = min(y_br, height)
        new_values = (x_tl, y_tl, x_br, y_br)
        self._validate_and_set_values(*new_values)
        return self

    def scale(self, factor: float) -> "BoundingBox":
        if factor <= 0:
            raise ValueError("Scaling 'factor' must be a positive value.")
        x_tl, y_tl, x_br, y_br = self.raw_values
        w, h = x_br - x_tl, y_br - y_tl
        x_c, y_c = x_tl + w / 2, y_tl + h / 2

        # Apply sqrt for both w and h to scale w.r.t area.
        w *= sqrt(factor)
        h *= sqrt(factor)
        new_values = (x_c - w / 2, y_c - h / 2, x_c + w / 2, y_c + h / 2)
        self._validate_and_set_values(*new_values)
        return self

    def shift(self, amount: Tuple[int, int]) -> "BoundingBox":
        x_tl, y_tl, x_br, y_br = self.raw_values
        horizontal_shift, vertical_shift = amount

        new_values = (x_tl + horizontal_shift, y_tl + vertical_shift, x_br + horizontal_shift, y_br + vertical_shift)
        self._validate_and_set_values(*new_values)
        return self

    def _to_bbox_type(self, name: str, return_values: bool) -> BaseBoundingBox:
        return load_bbox(
            name,
            values=self.raw_values,
            image_size=self.image_size,
            return_values=return_values,
            from_voc=True,
            strict=self.strict,
        )

    def to_albumentations(
        self, return_values: bool = False, **kwargs
    ) -> Union[Tuple[int, int, int, int], "BaseBoundingBox"]:
        return self._to_bbox_type("albumentations", return_values, **kwargs)

    def to_coco(self, return_values: bool = False, **kwargs) -> Union[Tuple[int, int, int, int], "BaseBoundingBox"]:
        return self._to_bbox_type("coco", return_values, **kwargs)

    def to_fiftyone(self, return_values: bool = False, **kwargs) -> Union[Tuple[int, int, int, int], "BaseBoundingBox"]:
        return self._to_bbox_type("fiftyone", return_values, **kwargs)

    def to_voc(self, return_values: bool = False, **kwargs) -> Union[Tuple[int, int, int, int], "BaseBoundingBox"]:
        return self._to_bbox_type("voc", return_values, **kwargs)

    def to_yolo(self, return_values: bool = False, **kwargs) -> Union[Tuple[int, int, int, int], "BaseBoundingBox"]:
        return self._to_bbox_type("yolo", return_values, **kwargs)

    @classmethod
    def from_voc(
        cls,
        x_tl: int,
        y_tl: int,
        x_br: int,
        y_br: int,
        image_size: Tuple[int, int] = None,
        strict: bool = True,
    ) -> "BaseBoundingBox":
        return load_bbox("voc", values=(x_tl, y_tl, x_br, y_br), image_size=image_size, strict=strict)

    @classmethod
    def from_albumentations(
        cls,
        x_tl: float,
        y_tl: float,
        x_br: float,
        y_br: float,
        image_size: Tuple[int, int] = None,
        strict: bool = False,
    ):
        return load_bbox("albumentations", values=(x_tl, y_tl, x_br, y_br), image_size=image_size, strict=strict)

    @classmethod
    def from_coco(cls, x_tl: int, y_tl: int, w: int, h: int, image_size: Tuple[int, int] = None, strict: bool = False):
        return load_bbox("coco", values=(x_tl, y_tl, w, h), image_size=image_size, strict=strict)

    @classmethod
    def from_fiftyone(
        cls, x_tl: float, y_tl: float, w: float, h: float, image_size: Tuple[int, int] = None, strict: bool = False
    ):
        return load_bbox("fiftyone", values=(x_tl, y_tl, w, h), image_size=image_size, strict=strict)

    @classmethod
    def from_yolo(
        cls, x_c: float, y_c: float, w: float, h: float, image_size: Tuple[int, int] = None, strict: bool = False
    ):
        return load_bbox("yolo", values=(x_c, y_c, w, h), image_size=image_size, strict=strict)
