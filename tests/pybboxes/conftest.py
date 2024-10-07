"""
Testing the package pybboxes.
Default image/bbox selected from the following source
https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/
"""
import inspect
import os
from typing import Optional

import numpy as np
import pytest

from tests.pybboxes import EXPECTED_OUTPUTS
from tests.utils import load_json


@pytest.fixture
def seed():
    return 42


@pytest.fixture(scope="package")
def image_size():
    return 640, 480


@pytest.fixture(scope="package")
def bbox_area():
    return 322 * 117  # w x h


@pytest.fixture(scope="package")
def albumentations_bbox():
    return [0.153125, 0.71875, 0.65625, 0.9625]


@pytest.fixture(scope="package")
def unnormalized_bbox_shift_amount():
    return (2, 2)


@pytest.fixture(scope="package")
def normalized_bbox_shift_amount():
    return (0.05, 0.03)


@pytest.fixture(scope="package")
def scale_factor():
    return 0.5


@pytest.fixture(scope="package")
def coco_bbox():
    return [98, 345, 322, 117]


@pytest.fixture(scope="package")
def fiftyone_bbox():
    return [0.153125, 0.71875, 0.503125, 0.24375]


@pytest.fixture(scope="package")
def voc_bbox():
    return [98, 345, 420, 462]


@pytest.fixture(scope="package")
def yolo_bbox():
    return [0.4046875, 0.840625, 0.503125, 0.24375]


@pytest.fixture
def multiple_bbox_shape():
    return 8, 3, 100, 2


@pytest.fixture
def expected_multiple_bbox_shape():
    return 8, 3, 100


@pytest.fixture
def multiple_albumentations_bboxes(multiple_bbox_shape, seed):
    np.random.seed(seed)
    a = np.random.uniform(0, 0.5, size=multiple_bbox_shape)
    b = np.random.uniform(0.5, 1, size=multiple_bbox_shape)
    return np.concatenate([a, b], -1)


@pytest.fixture
def multiple_coco_bboxes(multiple_bbox_shape, image_size, seed):
    np.random.seed(seed)
    w, h = image_size
    a = np.random.randint(0, min(w, h) - 30, size=multiple_bbox_shape)
    b = np.random.randint(1, 30, size=multiple_bbox_shape)
    return np.concatenate([a, b], -1)


@pytest.fixture
def multiple_fiftyone_bboxes(multiple_bbox_shape, seed):
    np.random.seed(seed)
    a = np.random.uniform(0, 0.8, size=multiple_bbox_shape)
    b = np.random.uniform(0, 0.2, size=multiple_bbox_shape)
    return np.concatenate([a, b], -1)


@pytest.fixture
def multiple_voc_bboxes(multiple_bbox_shape, image_size, seed):
    np.random.seed(seed)
    w, h = image_size
    cut = min(w, h) // 2
    a = np.random.randint(0, cut, size=multiple_bbox_shape)
    b = np.random.randint(cut, min(w, h), size=multiple_bbox_shape)
    return np.concatenate([a, b], -1)


@pytest.fixture
def multiple_yolo_bboxes(multiple_bbox_shape, seed):
    np.random.seed(seed)
    a = np.random.uniform(0, 0.6, size=multiple_bbox_shape)
    b = np.random.uniform(0, 0.2, size=multiple_bbox_shape)
    return np.concatenate([a, b], -1)


def get_expected_output(prefix: Optional[str] = None):
    def wrapper(fn, *args, **kwargs):
        module_name = os.path.basename(inspect.getfile(fn)).replace(".py", "")
        path = os.path.join(EXPECTED_OUTPUTS, prefix, f"{module_name}.json")
        test_name = fn.__name__.replace("output_", "")
        fn.output = load_json(path)[test_name]
        return fn

    if prefix is None:
        prefix = ""
    return wrapper
