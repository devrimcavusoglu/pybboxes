from typing import Sequence, Union

import numpy as np

IntegerBox: Union[Sequence[int], Sequence[Sequence[int]]]
FloatBox: Union[Sequence[float], Sequence[Sequence[float]]]


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
