from typing import Union

import numpy as np

from pybboxes.types.box_2d import IntegerBox, FloatBox


class Polygon:
	def __init__(self):
		self._points = []

	@property
	def points(self):
		return self._points

	def add(self, point: Union[IntegerBox, FloatBox]) -> None:
		point = np.array(point, dtype=float)
		self.points.append(point.tolist())

