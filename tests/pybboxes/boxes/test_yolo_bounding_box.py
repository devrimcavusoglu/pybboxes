import numpy as np
import pytest

from pybboxes import BoundingBox, YoloBoundingBox
from tests.utils import assert_almost_equal


@pytest.fixture(scope="module")
def yolo_bounding_box(yolo_bbox, image_size):
    return BoundingBox.from_yolo(*yolo_bbox, image_size=image_size)


@pytest.fixture(scope="module")
def yolo_oob_bounding_box():
    return [0.515625, 0.6614583333333334, 0.71875, 0.8854166666666666]


@pytest.fixture(scope="module")
def yolo_bounding_box2(yolo_bbox, image_size):
    np.random.seed(42)
    yolo_bbox2 = yolo_bbox + np.random.uniform(-0.05, 0.01, size=4)
    return BoundingBox.from_yolo(*yolo_bbox2, image_size=image_size)


@pytest.fixture(scope="function")
def yolo_area_computations_expected_output():
    return {
        "total_area": 72654,
        "union": 39314,
        "intersection": 33340,
        "iou": 0.8480439538078038,
        "ratio": 1.0770154373927958,
        "difference": 4334,
    }


def test_from_array(yolo_bbox, image_size):
    with pytest.warns(FutureWarning):
        yolo_box = YoloBoundingBox.from_array(yolo_bbox, image_size=image_size)

    assert yolo_box.is_oob is False


def test_shift(yolo_bounding_box, normalized_bbox_shift_amount):
    actual_output = yolo_bounding_box.shift(normalized_bbox_shift_amount)
    x_tl, y_tl, w, h = yolo_bounding_box.values
    desired = (x_tl + normalized_bbox_shift_amount[0], y_tl + normalized_bbox_shift_amount[1], w, h)

    print(actual_output, desired, "Yolo Bounding Box")

    assert_almost_equal(actual=actual_output.values, desired=desired)


def test_oob(yolo_oob_bounding_box, image_size):
    with pytest.raises(ValueError):
        BoundingBox.from_yolo(*yolo_oob_bounding_box, image_size=image_size)

    yolo_box = BoundingBox.from_yolo(*yolo_oob_bounding_box, image_size=image_size, strict=False)
    assert yolo_box.is_oob is True


# Conversions


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
