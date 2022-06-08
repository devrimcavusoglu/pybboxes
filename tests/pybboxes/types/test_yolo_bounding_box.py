import numpy as np
import pytest

from pybboxes import BoundingBox
from tests.utils import assert_almost_equal


@pytest.fixture(scope="module")
def yolo_bounding_box(yolo_bbox, image_size):
    return BoundingBox.from_yolo(*yolo_bbox, image_size=image_size)


@pytest.fixture(scope="module")
def yolo_bounding_box2(yolo_bbox, image_size):
    np.random.seed(42)
    yolo_bbox2 = yolo_bbox + np.random.uniform(-0.05, 0.05, size=4)
    return BoundingBox.from_yolo(*yolo_bbox2, image_size=image_size)


@pytest.fixture(scope="function")
def yolo_area_computations_expected_output():
    return {
        "total_area": 78330,
        "union": 46919,
        "intersection": 31411,
        "iou": 0.6694729214177625,
        "ratio": 0.9266528925619835,
        "difference": 6263,
    }


def test_to_albumentations(yolo_bounding_box, albumentations_bbox):
    yolo2albumentations_bbox = yolo_bounding_box.to_albumentations()
    assert_almost_equal(actual=list(yolo2albumentations_bbox.values), desired=albumentations_bbox)


def test_to_coco(yolo_bounding_box, coco_bbox):
    yolo2coco_bbox = yolo_bounding_box.to_coco()
    assert_almost_equal(actual=list(yolo2coco_bbox.values), desired=coco_bbox)


def test_to_fiftyone(yolo_bounding_box, fiftyone_bbox):
    yolo2fiftyone_bbox = yolo_bounding_box.to_fiftyone()
    assert_almost_equal(actual=list(yolo2fiftyone_bbox.values), desired=fiftyone_bbox)


def test_to_voc(yolo_bounding_box, voc_bbox):
    yolo2voc_bbox = yolo_bounding_box.to_voc()
    assert_almost_equal(actual=list(yolo2voc_bbox.values), desired=voc_bbox)


def test_area_computations(yolo_bounding_box, yolo_bounding_box2, yolo_area_computations_expected_output):
    actual_output = {
        "total_area": yolo_bounding_box.area + yolo_bounding_box2.area,
        "union": yolo_bounding_box + yolo_bounding_box2,
        "intersection": yolo_bounding_box * yolo_bounding_box2,
        "iou": yolo_bounding_box.iou(yolo_bounding_box2),
        "ratio": yolo_bounding_box / yolo_bounding_box2,
        "difference": yolo_bounding_box - yolo_bounding_box2,
    }
    assert_almost_equal(actual=actual_output, desired=yolo_area_computations_expected_output)
