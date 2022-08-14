from typing import Tuple, Union

from pybboxes.boxes.base import BaseBoundingBox
from pybboxes.boxes.bbox import BoundingBox


class CocoBoundingBox(BaseBoundingBox):
    def __init__(
        self,
        x_tl: int,
        y_tl: int,
        w: int,
        h: int,
        image_size: Tuple[int, int] = None,
        strict: bool = False,
    ):
        super(CocoBoundingBox, self).__init__(x_tl, y_tl, w, h, image_size=image_size, strict=strict)

    def _correct_value_types(self, *values):
        return tuple([round(val) for val in values])

    def _validate_values(self, *values):
        image_width, image_height = self.image_size

        x_tl, y_tl, w, h = values
        if w <= 0 or h <= 0:
            raise ValueError("Given width and height must be greater than 0.")
        elif self.strict and (x_tl < 0 or y_tl < 0):
            raise ValueError("Given top-left point is out of bounds.")
        elif (image_width is not None and x_tl + w > image_width) or (
            image_width is not None and y_tl + h > image_height
        ):
            if self.strict:
                raise ValueError(
                    "Given bounding box values is out of bounds. "
                    "To silently skip out of bounds cases pass 'strict=False'."
                )
            self._is_oob = True
        elif not self.is_image_size_null():
            self._is_oob = False

    def to_voc(self, return_values: bool = False) -> Union[Tuple[int, int, int, int], "BoundingBox"]:
        x_tl, y_tl, w, h = self.values
        x_br = x_tl + w
        y_br = y_tl + h
        if return_values:
            return x_tl, y_tl, x_br, y_br
        return BoundingBox(x_tl, y_tl, x_br, y_br, image_size=self.image_size, strict=self.strict)

    @classmethod
    def from_voc(
        cls,
        x_tl: int,
        y_tl: int,
        x_br: int,
        y_br: int,
        image_size: Tuple[int, int] = None,
        strict: bool = True,
    ) -> "CocoBoundingBox":
        w = x_br - x_tl
        h = y_br - y_tl
        return cls(x_tl, y_tl, w, h, image_size=image_size, strict=strict)
