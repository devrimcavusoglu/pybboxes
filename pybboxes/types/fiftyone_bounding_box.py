from typing import Tuple, Union

from pybboxes.types.base import BaseBoundingBox
from pybboxes.types.bbox import BoundingBox


class FiftyoneBoundingBox(BaseBoundingBox):
    def __init__(
        self,
        x_tl: float,
        y_tl: float,
        w: float,
        h: float,
        image_size: Tuple[int, int],
        strict: bool = True,
    ):
        super(FiftyoneBoundingBox, self).__init__(x_tl, y_tl, w, h, image_size=image_size, strict=strict)

    def _validate_values(self, *values):
        x_tl, y_tl, w, h = values
        if not 0 < w <= 1 or not 0 < h <= 1:
            raise ValueError("Given width and height must be in the range (0,1].")
        elif self.strict and (not 0 <= x_tl < 1 or not 0 <= y_tl < 1):
            raise ValueError("Given bounding box values is out of bounds.")

    def to_voc(self, return_values: bool = False) -> Union[Tuple[int, int, int, int], "BoundingBox"]:
        x_tl, y_tl, w, h = self.values
        image_width, image_height = self.image_size
        x_tl *= image_width
        y_tl *= image_height
        w *= image_width
        h *= image_height
        x_br = x_tl + w
        y_br = y_tl + h
        if return_values:
            return x_tl, y_tl, x_br, y_br
        return BoundingBox(x_tl, y_tl, x_br, y_br, image_size=self.image_size, strict=self.strict)

    def shift(self, amount: Tuple[float, float]) -> "FiftyoneBoundingBox":
        """Returns a new bounding box shifted by the given thresholds. The new
        bounding box has same image shape, and other properties as the current
        object.

        Parameters
        ----------
        amount: Tuple[float, float]
            The amount to shift the bounding box. The first value is the
                amount to shift the x-coordinate, and the second value is the
                amount to shift the y-coordinate.

        Returns
        -------
        FiftyoneBoundingBox
            The new bounding box.
        """
        x_tl, y_tl, w, h = self.values
        horizontal_shift, vertical_shift = amount

        return FiftyoneBoundingBox(
            x_tl + horizontal_shift,
            y_tl + vertical_shift,
            w,
            h,
            self.image_size,
            self.strict,
        )

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
