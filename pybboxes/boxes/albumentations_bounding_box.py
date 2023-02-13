from typing import Tuple, Union

from pybboxes.boxes.base import BaseBoundingBox
from pybboxes.boxes.bbox import BoundingBox


class AlbumentationsBoundingBox(BaseBoundingBox):
    def __init__(
        self,
        x_tl: float,
        y_tl: float,
        x_br: float,
        y_br: float,
        image_size: Tuple[int, int],
        strict: bool = False,
    ):
        super(AlbumentationsBoundingBox, self).__init__(x_tl, y_tl, x_br, y_br, image_size=image_size, strict=strict)

    def _validate_values(self, x_tl, y_tl, x_br, y_br):
        if not (0 <= x_tl < x_br <= 1) or not (0 <= y_tl < y_br <= 1):
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
        x_tl, y_tl, x_br, y_br = self.values
        image_width, image_height = self.image_size
        x_tl = round(x_tl * image_width)
        y_tl = round(y_tl * image_height)
        x_br = round(x_br * image_width)
        y_br = round(y_br * image_height)
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
        strict: bool = False,
    ) -> "AlbumentationsBoundingBox":
        if image_size is None:
            raise ValueError("AlbumentationsBoundingBox requires `image_size` to scale the box values.")
        image_width, image_height = image_size
        x_tl /= image_width
        y_tl /= image_height
        x_br /= image_width
        y_br /= image_height
        return cls(x_tl, y_tl, x_br, y_br, image_size=image_size, strict=strict)
