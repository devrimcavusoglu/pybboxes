from abc import ABC, abstractmethod
from typing import List, Tuple, Union

import numpy as np

from pybboxes.types.box_2d import Box

NORMALIZED_BOXES = ["albumentations", "fiftyone", "yolo"]


class BaseBoundingBox(Box, ABC):
    def __init__(
        self,
        v1: Union[int, float],
        v2: Union[int, float],
        v3: Union[int, float],
        v4: Union[int, float],
        image_size: Tuple[int, int] = None,
        strict: bool = False,
    ):
        self._image_size = image_size
        self.strict = strict
        self._is_oob = None
        self._validate_and_set_values(v1, v2, v3, v4)
        voc_values = self.to_voc(return_values=True)
        super(BaseBoundingBox, self).__init__(*voc_values)

    def __repr__(self):
        image_width, image_height = self.image_size
        str_vals = " ".join([f"{v:.4f}" if isinstance(v, float) else str(v) for v in self.values])
        return f"<[{str_vals}] ({self.width}x{self.height}) | Image: " f"({image_width or '?'}x{image_height or '?'})>"

    @property
    def is_oob(self) -> Union[bool, None]:
        """
        Whether the box is OOB (Out-of-bounds).

        Returns:
            None -> unknown. False -> Not OOB. True -> OOB.
        """
        return self._is_oob

    @property
    def image_size(self):
        if self._image_size is not None:
            return self._image_size
        else:
            return None, None

    @image_size.setter
    def image_size(self, image_size: Tuple[int, int]):
        self._image_size = image_size

    def is_image_size_null(self):
        if self.image_size == (None, None):
            return True
        return False

    @property
    def values(self) -> Tuple:
        return self._values

    def _correct_value_types(self, *values) -> Tuple:
        return values

    @abstractmethod
    def _validate_values(self, *values):
        pass

    def _set_values(self, *values):
        """
        This method is intended to be "final", and should not be overridden in child classes.
        """
        self._values = values

    def _validate_and_set_values(self, *values) -> None:
        """
        Validate and sets given values if validation is successful.
        """
        self.raw_values = values
        values = self._correct_value_types(*values)
        self._validate_values(*values)
        self._set_values(*values)

    def to_albumentations(self, return_values: bool = False) -> Union[Tuple[int, int, int, int], "BaseBoundingBox"]:
        return self.to_voc().to_albumentations(return_values)

    def to_coco(self, return_values: bool = False) -> Union[Tuple[int, int, int, int], "BaseBoundingBox"]:
        return self.to_voc().to_coco(return_values)

    def to_fiftyone(self, return_values: bool = False) -> Union[Tuple[int, int, int, int], "BaseBoundingBox"]:
        return self.to_voc().to_fiftyone(return_values)

    @abstractmethod
    def to_voc(self, return_values: bool = False) -> Union[Tuple[int, int, int, int], "BaseBoundingBox"]:
        pass

    def to_yolo(self, return_values: bool = False) -> Union[Tuple[int, int, int, int], "BaseBoundingBox"]:
        return self.to_voc().to_yolo(return_values)

    @property
    def name(self):
        return self.__class__.__name__.lower().replace("boundingbox", "")

    def _generic_operation(self, op: str, *args, **kwargs) -> None:
        refined_box = self.to_voc()
        box_op = getattr(refined_box, op)
        refined_box = box_op(*args, **kwargs)
        box_conversion = getattr(refined_box, f"to_{self.name}")
        refined_box = box_conversion()

        self.__init__(*refined_box.values, image_size=self.image_size, strict=self.strict)

    def clamp(self) -> "BaseBoundingBox":
        """
        Clamps the box with respect to the image borders. If the box is not OOB, does nothing.
        """
        self._generic_operation("clamp")
        return self

    def scale(self, factor: float) -> "BaseBoundingBox":
        self._generic_operation("scale", factor)
        return self

    def shift(self, amount: Tuple) -> "BaseBoundingBox":
        """
        Perform a shift operation on the bounding box inplace.

        Args:
            amount: The amount to shift the bounding box. The first value is the
                amount to shift the x-coordinate, and the second value is the
                amount to shift the y-coordinate.
        """
        if self.name in NORMALIZED_BOXES:
            width, height = self.image_size
            amount = (amount[0] * width, amount[1] * height)
        self._generic_operation("shift", amount)
        return self

    @classmethod
    @abstractmethod
    def from_voc(
        cls,
        x_tl: int,
        y_tl: int,
        x_br: int,
        y_br: int,
        image_size: Tuple[int, int] = None,
        strict: bool = True,
    ) -> "BaseBoundingBox":
        pass

    @classmethod
    def from_array_vectorize(cls, ar: np.ndarray):
        constructor = cls.from_array
        vconstructor = np.vectorize(constructor)
        return vconstructor(ar)

    @classmethod
    def from_array(cls, ar: Union[Tuple, List, np.ndarray], **kwargs) -> Union[np.ndarray, "BaseBoundingBox"]:
        """
        Takes input values containing at least a single bbox values. Input can be multidimensional
        array as long as the last dimension (-1) has length of 4, i.e for any array as input, the shape
        should look like (x,y,z,4) and the output is of shape (x,y,z).

        Args:
            ar: Input values as a tuple or array. If the input is an array, the dimension is preserved as is
                and each bounding box values is converted to the `BoundingBox` object.
            **kwargs: Additional keyword arguments for construction, see :py:meth:`BoundingBox.__init__`

        Notes:
            This method is intended to be "final", and should not be overridden in child classes.

        Returns:
            Either a `BoundingBox` object constructed from input values or list of `BoundingBox` objects
            as an array.
        """
        if not isinstance(ar, np.ndarray):
            ar = np.array(ar)
        if ar.shape[-1] != 4:
            raise ValueError(f"Given input array must have bounding box values at dim -1 as 4, got shape {ar.shape}.")
        if ar.ndim == 1:
            return cls(*ar, **kwargs)
        vf = np.vectorize(cls.from_array, signature="(n) -> ()", excluded={"image_size", "strict"})
        return vf(ar, **kwargs)
