import numpy as np
import pytest

from pybboxes import BoundingBox, FiftyoneBoundingBox
from tests.utils import assert_almost_equal


@pytest.fixture(scope="function")
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


@pytest.fixture()
def scaled_fiftyone_box():
    return 0.22680595035775913, 0.7544463610428895, 0.35576309928448174, 0.17235727791422092


@pytest.fixture(scope="function")
def fiftyone_area_computations_expected_output():
    return {
        "total_area": 72972,
        "union": 39672,
        "intersection": 33300,
        "iou": 0.8393829401088929,
        "ratio": 1.0673125956144824,
        "difference": 4374,
    }


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


def test_from_array(fiftyone_bbox, image_size):
    with pytest.warns(FutureWarning):
        fo_box = FiftyoneBoundingBox.from_array(fiftyone_bbox, image_size=image_size)

    assert fo_box.is_oob is False


@pytest.mark.parametrize(
    "box_values,expected_out",
    [
        ((0.15625, 0.21875, 0.875, 0.9), (0.15625, 0.21875, 0.84375, 0.78125)),
        ((-0.05, -0.05, 0.5, 0.5), (0.0, 0.0, 0.45, 0.45)),
        ((0.3, 0.4, 1, 0.7), (0.3, 0.4, 0.7, 0.6)),
    ],
)
def test_clamp(box_values, expected_out, image_size):
    fo_box = FiftyoneBoundingBox(*box_values, image_size=image_size)
    fo_box.clamp()

    assert_almost_equal(actual=fo_box.values, desired=expected_out, ignore_numeric_type_changes=True)


def test_scale(fiftyone_bounding_box, scaled_fiftyone_box, scale_factor):
    _, _, w, h = fiftyone_bounding_box.values
    image_width, image_height = fiftyone_bounding_box.image_size
    w, h = w * image_width, h * image_height

    fiftyone_bounding_box.scale(scale_factor)

    assert_almost_equal(
        actual=fiftyone_bounding_box.values, desired=scaled_fiftyone_box, ignore_numeric_type_changes=True
    )

    actual_area = fiftyone_bounding_box.area
    desired_area = w * h * scale_factor
    assert actual_area - desired_area < 10**2


def test_shift(fiftyone_bounding_box, normalized_bbox_shift_amount):
    x_tl, y_tl, w, h = fiftyone_bounding_box.values
    desired = (x_tl + normalized_bbox_shift_amount[0], y_tl + normalized_bbox_shift_amount[1], w, h)
    actual_output = fiftyone_bounding_box.shift(normalized_bbox_shift_amount)

    assert_almost_equal(actual=actual_output.values, desired=desired)


def test_oob(fiftyone_oob_bounding_box, image_size):
    with pytest.raises(ValueError):
        BoundingBox.from_fiftyone(*fiftyone_oob_bounding_box, image_size=image_size, strict=True)

    fo_box = BoundingBox.from_fiftyone(*fiftyone_oob_bounding_box, image_size=image_size)
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
