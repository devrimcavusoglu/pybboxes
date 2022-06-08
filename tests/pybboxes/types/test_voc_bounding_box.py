import numpy as np
import pytest

from pybboxes import BoundingBox
from tests.utils import assert_almost_equal


@pytest.fixture(scope="module")
def voc_bounding_box(voc_bbox, image_size):
    return BoundingBox.from_voc(*voc_bbox, image_size=image_size)


@pytest.fixture(scope="module")
def voc_bounding_box2(voc_bbox, image_size):
    np.random.seed(42)
    voc_bbox2 = voc_bbox + np.random.randint(-5, 5, size=4)
    return BoundingBox.from_voc(*voc_bbox2, image_size=image_size)


@pytest.fixture(scope="function")
def voc_area_computations_expected_output():
    return {
        "total_area": 75788,
        "union": 38552,
        "intersection": 37236,
        "iou": 0.9658642871965138,
        "ratio": 0.9884556855748544,
        "difference": 438,
    }


def test_to_albumentations(voc_bounding_box, albumentations_bbox):
    voc2albumentations_bbox = voc_bounding_box.to_albumentations()
    assert_almost_equal(actual=list(voc2albumentations_bbox.values), desired=albumentations_bbox)


def test_to_fiftyone(voc_bounding_box, fiftyone_bbox):
    voc2fiftyone_bbox = voc_bounding_box.to_fiftyone()
    assert_almost_equal(actual=list(voc2fiftyone_bbox.values), desired=fiftyone_bbox)


def test_to_coco(voc_bounding_box, coco_bbox):
    voc2coco_bbox = voc_bounding_box.to_coco()
    assert_almost_equal(actual=list(voc2coco_bbox.values), desired=coco_bbox)


def test_to_yolo(voc_bounding_box, yolo_bbox):
    voc2yolo_bbox = voc_bounding_box.to_yolo()
    assert_almost_equal(actual=list(voc2yolo_bbox.values), desired=yolo_bbox)


def test_area_computations(voc_bounding_box, voc_bounding_box2, voc_area_computations_expected_output):
    actual_output = {
        "total_area": voc_bounding_box.area + voc_bounding_box2.area,
        "union": voc_bounding_box + voc_bounding_box2,
        "intersection": voc_bounding_box * voc_bounding_box2,
        "iou": voc_bounding_box.iou(voc_bounding_box2),
        "ratio": voc_bounding_box / voc_bounding_box2,
        "difference": voc_bounding_box - voc_bounding_box2,
    }
    assert_almost_equal(actual=actual_output, desired=voc_area_computations_expected_output)
