from typing import Tuple, Union

from pybboxes.boxes.base import BaseBoundingBox
from pybboxes.boxes.bbox import BoundingBox


class FiftyoneBoundingBox(BaseBoundingBox):
    def __init__(
        self,
        x_tl: float,
        y_tl: float,
        w: float,
        h: float,
        image_size: Tuple[int, int],
        strict: bool = False,
    ):
        super(FiftyoneBoundingBox, self).__init__(x_tl, y_tl, w, h, image_size=image_size, strict=strict)

    def _validate_values(self, *values):
        x_tl, y_tl, w, h = values
        if not 0 < w <= 1 or not 0 < h <= 1:
            raise ValueError("Given width and height must be in the range (0,1].")
        elif not 0 <= x_tl < x_tl + w <= 1 or not 0 <= y_tl < y_tl + h <= 1:
            if self.strict:
                raise ValueError(
                    "Given bounding box values is out of bounds. "
                    "To silently skip out of bounds cases pass 'strict=False'."
                )
            self._is_oob = True
        else:
            self._is_oob = False

    def to_voc(self, return_values: bool = False) -> Union[Tuple[int, int, int, int], "BoundingBox"]:
        if self.is_image_size_null():
            raise ValueError("'image_size' is required for conversion.")
        x_tl, y_tl, w, h = self.values
        image_width, image_height = self.image_size
        x_tl *= image_width
        y_tl *= image_height
        w *= image_width
        h *= image_height
        x_br = x_tl + w
        y_br = y_tl + h
        x_tl, y_tl, x_br, y_br = round(x_tl), round(y_tl), round(x_br), round(y_br)
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
    ) -> "FiftyoneBoundingBox":
        if image_size is None:
            raise ValueError("YoloBounding box requires `image_size` to normalize the box values.")
        image_width, image_height = image_size
        w = x_br - x_tl
        h = y_br - y_tl

        x_tl /= image_width
        y_tl /= image_height
        w /= image_width
        h /= image_height
        return cls(x_tl, y_tl, w, h, image_size=image_size, strict=strict)
