import numpy as np
import pytest

from pybboxes import BoundingBox, YoloBoundingBox
from tests.utils import assert_almost_equal


@pytest.fixture(scope="function")
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


@pytest.fixture()
def scaled_yolo_box():
    return 0.4046875, 0.840625, 0.3557630992844818, 0.17235727791422098


@pytest.fixture(scope="function")
def yolo_area_computations_expected_output():
    return {
        "total_area": 72654,
        "union": 39434,
        "intersection": 33220,
        "iou": 0.8424202464878024,
        "ratio": 1.0770154373927958,
        "difference": 4454,
    }


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


def test_from_array(yolo_bbox, image_size):
    with pytest.warns(FutureWarning):
        yolo_box = YoloBoundingBox.from_array(yolo_bbox, image_size=image_size)

    assert yolo_box.is_oob is False


@pytest.mark.parametrize(
    "box_values,expected_out",
    [
        ((0.15625, 0.21875, 0.875, 0.9), (0.296875, 0.334375, 0.59375, 0.66875)),
        ((-0.05, -0.05, 0.5, 0.5), (0.1, 0.1, 0.2, 0.2)),
        ((0.3, 0.4, 0.7, 0.7), (0.325, 0.4, 0.65, 0.7)),
    ],
)
def test_clamp(box_values, expected_out, image_size):
    yolo_box = YoloBoundingBox(*box_values, image_size=image_size)
    yolo_box.clamp()

    assert_almost_equal(actual=yolo_box.values, desired=expected_out, ignore_numeric_type_changes=True)


def test_shift(yolo_bounding_box, normalized_bbox_shift_amount):
    x_c, y_c, w, h = yolo_bounding_box.values
    desired = (x_c + normalized_bbox_shift_amount[0], y_c + normalized_bbox_shift_amount[1], w, h)
    actual_output = yolo_bounding_box.shift(normalized_bbox_shift_amount)

    assert_almost_equal(actual=actual_output.values, desired=desired)


def test_scale(yolo_bounding_box, scaled_yolo_box, scale_factor):
    _, _, w, h = yolo_bounding_box.values
    image_width, image_height = yolo_bounding_box.image_size
    w, h = w * image_width, h * image_height

    yolo_bounding_box.scale(scale_factor)

    assert_almost_equal(actual=yolo_bounding_box.values, desired=scaled_yolo_box, ignore_numeric_type_changes=True)

    actual_area = yolo_bounding_box.area
    desired_area = w * h * scale_factor
    assert actual_area - desired_area < 10**2


def test_oob(yolo_oob_bounding_box, image_size):
    with pytest.raises(ValueError):
        BoundingBox.from_yolo(*yolo_oob_bounding_box, image_size=image_size, strict=True)

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
