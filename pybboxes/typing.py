from typing import Sequence, Tuple, Union

import numpy as np

GenericBboxType = Union[
    np.ndarray, Tuple[float, float, float, float], Tuple[int, int, int, int], Sequence[float], Sequence[int]
]
BboxType = Union[Tuple[float, float, float, float], Tuple[int, int, int, int]]


def generic_to_strict(bbox: GenericBboxType, dtype: type = float) -> BboxType:
    bbox = np.array(bbox)
    if dtype == int:
        bbox = bbox.round().astype(int)

    return tuple(bbox.tolist())
