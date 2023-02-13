from typing import Tuple, Union

from pybboxes.boxes.base import BaseBoundingBox
from pybboxes.boxes.bbox import BoundingBox


class VocBoundingBox(BaseBoundingBox):
    """
    Alias for the VOC style bounding box.
    """

    def __init__(
        self,
        x_tl: int,
        y_tl: int,
        x_br: int,
        y_br: int,
        image_size: Tuple[int, int],
        strict: bool = False,
    ):
        super(VocBoundingBox, self).__init__(x_tl, y_tl, x_br, y_br, image_size=image_size, strict=strict)

    def _correct_value_types(self, *values):
        return tuple([round(val) for val in values])

    def _validate_values(self, x_tl, y_tl, x_br, y_br):
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

    def to_voc(self, return_values: bool = False) -> Union[Tuple[int, int, int, int], "BoundingBox"]:
        x_tl, y_tl, x_br, y_br = self.values
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
    ) -> "VocBoundingBox":
        return cls(x_tl, y_tl, x_br, y_br, image_size=image_size, strict=strict)
