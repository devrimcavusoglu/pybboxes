import numpy as np
import pytest

from pybboxes import BoundingBox
from tests.utils import assert_almost_equal


@pytest.fixture(scope="module")
def coco_bounding_box(coco_bbox, image_size):
    return BoundingBox.from_coco(*coco_bbox, image_size=image_size)


@pytest.fixture(scope="module")
def coco_bounding_box2(coco_bbox, image_size):
    np.random.seed(42)
    coco_bbox2 = coco_bbox + np.random.randint(-5, 5, size=4)
    return BoundingBox.from_coco(*coco_bbox2, image_size=image_size)


@pytest.fixture(scope="function")
def coco_area_computations_expected_output():
    return {
        "total_area": 75258,
        "union": 38664,
        "intersection": 36594,
        "iou": 0.9464618249534451,
        "ratio": 1.0023946360153257,
        "difference": 1080,
    }


def test_to_coco(coco_bounding_box, albumentations_bbox):
    coco2albumentations_bbox = coco_bounding_box.to_albumentations()
    assert_almost_equal(actual=list(coco2albumentations_bbox.values), desired=albumentations_bbox)


def test_to_fiftyone(coco_bounding_box, fiftyone_bbox):
    coco2fiftyone_bbox = coco_bounding_box.to_fiftyone()
    assert_almost_equal(actual=list(coco2fiftyone_bbox.values), desired=fiftyone_bbox)


def test_to_voc(coco_bounding_box, voc_bbox):
    coco2voc_bbox = coco_bounding_box.to_voc()
    assert_almost_equal(actual=list(coco2voc_bbox.values), desired=voc_bbox)


def test_to_yolo(coco_bounding_box, yolo_bbox):
    coco2yolo_bbox = coco_bounding_box.to_yolo()
    assert_almost_equal(actual=list(coco2yolo_bbox.values), desired=yolo_bbox)


def test_area_computations(coco_bounding_box, coco_bounding_box2, coco_area_computations_expected_output):
    actual_output = {
        "total_area": coco_bounding_box.area + coco_bounding_box2.area,
        "union": coco_bounding_box + coco_bounding_box2,
        "intersection": coco_bounding_box * coco_bounding_box2,
        "iou": coco_bounding_box.iou(coco_bounding_box2),
        "ratio": coco_bounding_box / coco_bounding_box2,
        "difference": coco_bounding_box - coco_bounding_box2,
    }
    assert_almost_equal(actual=actual_output, desired=coco_area_computations_expected_output)
