"""
Testing the package pybboxes.
Default image/bbox selected from the following source
https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/
"""
import inspect
import os
from typing import Optional

import pytest

from tests.pybboxes import EXPECTED_OUTPUTS
from tests.utils import load_json


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
