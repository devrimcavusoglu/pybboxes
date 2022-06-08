import numpy as np
import pytest

from pybboxes import BoundingBox
from tests.utils import assert_almost_equal


@pytest.fixture(scope="module")
def fiftyone_bounding_box(fiftyone_bbox, image_size):
    return BoundingBox.from_fiftyone(*fiftyone_bbox, image_size=image_size)


@pytest.fixture(scope="module")
def fiftyone_bounding_box2(fiftyone_bbox, image_size):
    np.random.seed(42)
    fiftyone_bbox2 = fiftyone_bbox + np.random.uniform(-0.05, 0.05, size=4)
    return BoundingBox.from_fiftyone(*fiftyone_bbox2, image_size=image_size)


@pytest.fixture(scope="function")
def fiftyone_area_computations_expected_output():
    return {
        "total_area": 78330,
        "union": 47623,
        "intersection": 30707,
        "iou": 0.6447934821409823,
        "ratio": 0.9266528925619835,
        "difference": 6967,
    }


def test_to_albumentations(fiftyone_bounding_box, albumentations_bbox):
    fiftyone2albumentations_bbox = fiftyone_bounding_box.to_albumentations()
    assert_almost_equal(actual=list(fiftyone2albumentations_bbox.values), desired=albumentations_bbox)


def test_to_coco(fiftyone_bounding_box, coco_bbox):
    fiftyone2coco_bbox = fiftyone_bounding_box.to_coco()
    assert_almost_equal(actual=list(fiftyone2coco_bbox.values), desired=coco_bbox)


def test_to_voc(fiftyone_bounding_box, voc_bbox):
    fiftyone2voc_bbox = fiftyone_bounding_box.to_voc()
    assert_almost_equal(actual=list(fiftyone2voc_bbox.values), desired=voc_bbox)


def test_to_yolo(fiftyone_bounding_box, yolo_bbox):
    fiftyone2yolo_bbox = fiftyone_bounding_box.to_yolo()
    assert_almost_equal(actual=list(fiftyone2yolo_bbox.values), desired=yolo_bbox)


def test_area_computations(fiftyone_bounding_box, fiftyone_bounding_box2, fiftyone_area_computations_expected_output):
    actual_output = {
        "total_area": fiftyone_bounding_box.area + fiftyone_bounding_box2.area,
        "union": fiftyone_bounding_box + fiftyone_bounding_box2,
        "intersection": fiftyone_bounding_box * fiftyone_bounding_box2,
        "iou": fiftyone_bounding_box.iou(fiftyone_bounding_box2),
        "ratio": fiftyone_bounding_box / fiftyone_bounding_box2,
        "difference": fiftyone_bounding_box - fiftyone_bounding_box2,
    }
    assert_almost_equal(actual=actual_output, desired=fiftyone_area_computations_expected_output)
