from typing import Sequence, Tuple, Union

import numpy as np

GenericBboxType = Union[
    np.ndarray, Tuple[float, float, float, float], Tuple[int, int, int, int], Sequence[float], Sequence[int]
]
BboxType = Union[Tuple[float, float, float, float], Tuple[int, int, int, int]]
