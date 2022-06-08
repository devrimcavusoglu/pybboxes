from abc import ABC, abstractmethod
from typing import List, Tuple, Union

import numpy as np


class Box:
    def __init__(self, x_tl: int, y_tl: int, x_br: int, y_br: int):
        self.x_tl = x_tl
        self.y_tl = y_tl
        self.x_br = x_br
        self.y_br = y_br

    def __add__(self, other: "Box") -> int:
        return self.union(other)

    def __sub__(self, other: "Box") -> int:
        return int(self.area - self.intersection(other))

    def __mul__(self, other: "Box") -> int:
        return self.intersection(other)

    def __truediv__(self, other: "Box") -> float:
        return self.area / other.area

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def height(self) -> int:
        return int(self.y_br - self.y_tl)

    @property
    def width(self) -> int:
        return int(self.x_br - self.x_tl)

    def intersection(self, other: "Box") -> int:
        x_tl, y_tl = np.maximum((self.x_tl, self.y_tl), (other.x_tl, other.y_tl))
        x_br, y_br = np.minimum((self.x_br, self.y_br), (other.x_br, other.y_br))
        if x_tl >= x_br or y_tl >= y_br:
            return 0
        intersection_width = x_br - x_tl
        intersection_height = y_br - y_tl
        return int(intersection_width * intersection_height)

    def union(self, other: "Box") -> int:
        return int(self.area + other.area - self.intersection(other))

    def iou(self, other: "Box") -> float:
        return self.intersection(other) / self.union(other)


class BaseBoundingBox(Box, ABC):
    def __init__(
        self,
        v1: Union[int, float],
        v2: Union[int, float],
        v3: Union[int, float],
        v4: Union[int, float],
        image_size: Tuple[int, int] = None,
        strict: bool = True,
    ):
        self._image_size = image_size
        self.strict = strict
        v1, v2, v3, v4 = self._correct_value_types(v1, v2, v3, v4)
        self._validate_values(v1, v2, v3, v4)
        self.__set_values(v1, v2, v3, v4)
        voc_values = self.to_voc(return_values=True)
        super(BaseBoundingBox, self).__init__(*voc_values)

    def __repr__(self):
        image_width, image_height = self.image_size
        str_vals = " ".join([f"{v:.4f}" if isinstance(v, float) else str(v) for v in self.values])
        return f"<[{str_vals}] ({self.width}x{self.height}) | Image: " f"({image_width or '?'}x{image_height or '?'})>"

    @property
    def image_size(self):
        if self._image_size is not None:
            return self._image_size
        else:
            return None, None

    @image_size.setter
    def image_size(self, image_size: Tuple[int, int]):
        self._image_size = image_size

    @property
    def values(self) -> Tuple:
        return self._values

    def _correct_value_types(self, *values) -> Tuple:
        return values

    @abstractmethod
    def _validate_values(self, *values):
        pass

    def __set_values(self, *values):
        """
        This method is intended to be "final", and should not be overridden in child classes.
        """
        self._values = values

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
    def from_array(cls, ar: Union[Tuple, List, np.ndarray], **kwargs):
        """
        This method is intended to be "final", and should not be overridden in child classes.
        """
        if len(ar) != 4:
            raise ValueError(f"Given array must be length of 4, got length {len(ar)}.")
        return cls(*ar, **kwargs)
