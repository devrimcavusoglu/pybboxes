import numpy as np
import pytest

from pybboxes import BoundingBox
from tests.utils import assert_almost_equal


@pytest.fixture(scope="module")
def albumentations_bounding_box(albumentations_bbox, image_size):
    return BoundingBox.from_albumentations(*albumentations_bbox, image_size=image_size)


@pytest.fixture(scope="module")
def albumentations_bounding_box2(albumentations_bbox, image_size):
    np.random.seed(42)
    albumentations_bbox2 = albumentations_bbox + np.random.uniform(-0.05, 0.05, size=4)
    return BoundingBox.from_albumentations(*albumentations_bbox2, image_size=image_size)


@pytest.fixture(scope="function")
def albumentations_area_computations_expected_output():
    return {
        "total_area": 72174,
        "union": 41584,
        "intersection": 30590,
        "iou": 0.7356194690265486,
        "ratio": 1.092,
        "difference": 7084,
    }


def test_to_coco(albumentations_bounding_box, coco_bbox):
    alb2coco_bbox = albumentations_bounding_box.to_coco()
    assert_almost_equal(actual=list(alb2coco_bbox.values), desired=coco_bbox)


def test_to_fiftyone(albumentations_bounding_box, fiftyone_bbox):
    alb2fiftyone_bbox = albumentations_bounding_box.to_fiftyone()
    assert_almost_equal(actual=list(alb2fiftyone_bbox.values), desired=fiftyone_bbox)


def test_to_voc(albumentations_bounding_box, voc_bbox):
    alb2voc_bbox = albumentations_bounding_box.to_voc()
    assert_almost_equal(actual=list(alb2voc_bbox.values), desired=voc_bbox)


def test_to_yolo(albumentations_bounding_box, yolo_bbox):
    alb2yolo_bbox = albumentations_bounding_box.to_yolo()
    assert_almost_equal(actual=list(alb2yolo_bbox.values), desired=yolo_bbox)


def test_area_computations(
    albumentations_bounding_box, albumentations_bounding_box2, albumentations_area_computations_expected_output
):
    actual_output = {
        "total_area": albumentations_bounding_box.area + albumentations_bounding_box2.area,
        "union": albumentations_bounding_box + albumentations_bounding_box2,
        "intersection": albumentations_bounding_box * albumentations_bounding_box2,
        "iou": albumentations_bounding_box.iou(albumentations_bounding_box2),
        "ratio": albumentations_bounding_box / albumentations_bounding_box2,
        "difference": albumentations_bounding_box - albumentations_bounding_box2,
    }
    assert_almost_equal(actual=actual_output, desired=albumentations_area_computations_expected_output)
