import numpy as np
import pytest

from pybboxes import BoundingBox, VocBoundingBox
from tests.utils import assert_almost_equal


@pytest.fixture(scope="function")
def voc_bounding_box(voc_bbox, image_size):
    return BoundingBox.from_voc(*voc_bbox, image_size=image_size)


@pytest.fixture(scope="module")
def voc_oob_bounding_box():
    return [100, 105, 560, 530]


@pytest.fixture(scope="module")
def voc_bounding_box2(voc_bbox, image_size):
    np.random.seed(42)
    voc_bbox2 = voc_bbox + np.random.randint(-5, 5, size=4)
    return BoundingBox.from_voc(*voc_bbox2, image_size=image_size)


@pytest.fixture()
def scaled_voc_box():
    return 145, 362, 373, 445


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


def test_from_array(voc_bbox, image_size):
    with pytest.warns(FutureWarning):
        voc_box = VocBoundingBox.from_array(voc_bbox, image_size=image_size)

    assert voc_box.is_oob is False


@pytest.mark.parametrize(
    "box_values,expected_out",
    [
        ((270, 350, 400, 450), (270, 350, 400, 450)),
        ((-50, -50, 342, 190), (0, 0, 342, 190)),
        ((153, 150, 490, 580), (153, 150, 490, 480)),
    ],
)
def test_clamp(box_values, expected_out, image_size):
    voc_box = VocBoundingBox(*box_values, image_size=image_size)
    voc_box.clamp()

    assert_almost_equal(actual=voc_box.values, desired=expected_out, ignore_numeric_type_changes=True)


def test_scale(voc_bounding_box, scaled_voc_box, scale_factor):
    x_tl, y_tl, x_br, y_br = voc_bounding_box.values
    w, h = (x_br - x_tl), (y_br - y_tl)

    voc_bounding_box.scale(scale_factor)

    assert_almost_equal(actual=voc_bounding_box.values, desired=scaled_voc_box, ignore_numeric_type_changes=True)

    actual_area = voc_bounding_box.area
    desired_area = w * h * scale_factor
    assert actual_area - desired_area < 10**2


def test_shift(voc_bounding_box, unnormalized_bbox_shift_amount):
    x_tl, y_tl, x_br, y_br = voc_bounding_box.values
    desired = (
        x_tl + unnormalized_bbox_shift_amount[0],
        y_tl + unnormalized_bbox_shift_amount[1],
        x_br + unnormalized_bbox_shift_amount[0],
        y_br + unnormalized_bbox_shift_amount[1],
    )
    actual_output = voc_bounding_box.shift(unnormalized_bbox_shift_amount)

    assert_almost_equal(actual=actual_output.values, desired=desired)


def test_oob(voc_oob_bounding_box, image_size):
    with pytest.raises(ValueError):
        BoundingBox.from_albumentations(*voc_oob_bounding_box, image_size=image_size, strict=True)

    voc_box = BoundingBox.from_albumentations(*voc_oob_bounding_box, image_size=image_size)
    assert voc_box.is_oob is True


# Conversions


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
