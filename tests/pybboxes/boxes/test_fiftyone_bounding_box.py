import numpy as np
import pytest

from pybboxes import BoundingBox, FiftyoneBoundingBox
from tests.utils import assert_almost_equal


@pytest.fixture(scope="module")
def fiftyone_bounding_box(fiftyone_bbox, image_size):
    return BoundingBox.from_fiftyone(*fiftyone_bbox, image_size=image_size)


@pytest.fixture(scope="module")
def fiftyone_oob_bounding_box():
    return [0.15625, 0.21875, 0.71875, 0.8854166666666666]


@pytest.fixture(scope="module")
def fiftyone_bounding_box2(fiftyone_bbox, image_size):
    np.random.seed(42)
    fiftyone_bbox2 = fiftyone_bbox + np.random.uniform(-0.05, 0.01, size=4)
    return BoundingBox.from_fiftyone(*fiftyone_bbox2, image_size=image_size)


@pytest.fixture(scope="function")
def fiftyone_area_computations_expected_output():
    return {
        "total_area": 72654,
        "union": 39528,
        "intersection": 33126,
        "iou": 0.8380388585306618,
        "ratio": 1.0770154373927958,
        "difference": 4548,
    }


def test_from_array(fiftyone_bbox, image_size):
    with pytest.warns(FutureWarning):
        fo_box = FiftyoneBoundingBox.from_array(fiftyone_bbox, image_size=image_size)

    assert fo_box.is_oob is False


def test_shift(fiftyone_bounding_box, normalized_bbox_shift_amount):
    actual_output = fiftyone_bounding_box.shift(normalized_bbox_shift_amount)
    x_tl, y_tl, w, h = fiftyone_bounding_box.values
    desired = (x_tl + normalized_bbox_shift_amount[0], y_tl + normalized_bbox_shift_amount[1], w, h)

    assert_almost_equal(actual=actual_output.values, desired=desired)


def test_oob(fiftyone_oob_bounding_box, image_size):
    with pytest.raises(ValueError):
        BoundingBox.from_fiftyone(*fiftyone_oob_bounding_box, image_size=image_size)

    fo_box = BoundingBox.from_fiftyone(*fiftyone_oob_bounding_box, image_size=image_size, strict=False)
    assert fo_box.is_oob is True


# Conversions


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
