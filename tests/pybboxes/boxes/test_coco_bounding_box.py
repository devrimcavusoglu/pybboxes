import numpy as np
import pytest

from pybboxes import BoundingBox, CocoBoundingBox
from tests.utils import assert_almost_equal


@pytest.fixture(scope="function")
def coco_bounding_box(coco_bbox, image_size):
    return BoundingBox.from_coco(*coco_bbox, image_size=image_size)


@pytest.fixture(scope="module")
def coco_oob_bounding_box():
    return [100, 105, 460, 425]


@pytest.fixture(scope="module")
def coco_bounding_box2(coco_bbox, image_size):
    np.random.seed(42)
    coco_bbox2 = coco_bbox + np.random.randint(-5, 5, size=4)
    return BoundingBox.from_coco(*coco_bbox2, image_size=image_size)


@pytest.fixture()
def scaled_coco_box():
    return 145, 362, 228, 83


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


def test_from_array(coco_bbox, image_size):
    with pytest.warns(FutureWarning):
        coco_box = CocoBoundingBox.from_array(coco_bbox, image_size=image_size)

    assert coco_box.is_oob is False


@pytest.mark.parametrize(
    "box_values,expected_out",
    [
        ((270, 350, 400, 450), (270, 350, 370, 130)),
        ((-50, -50, 342, 190), (0, 0, 292, 140)),
        ((153, 150, 490, 580), (153, 150, 487, 330)),
    ],
)
def test_clamp(box_values, expected_out, image_size):
    coco_box = CocoBoundingBox(*box_values, image_size=image_size)
    coco_box.clamp()

    assert_almost_equal(actual=coco_box.values, desired=expected_out, ignore_numeric_type_changes=True)


def test_scale(coco_bounding_box, scaled_coco_box, scale_factor):
    _, _, w, h = coco_bounding_box.values

    coco_bounding_box.scale(scale_factor)

    assert_almost_equal(actual=coco_bounding_box.values, desired=scaled_coco_box, ignore_numeric_type_changes=True)

    actual_area = coco_bounding_box.area
    desired_area = w * h * scale_factor
    assert actual_area - desired_area < 10**2


def test_shift(coco_bounding_box, unnormalized_bbox_shift_amount):
    x_tl, y_tl, w, h = coco_bounding_box.values
    desired = (x_tl + unnormalized_bbox_shift_amount[0], y_tl + unnormalized_bbox_shift_amount[1], w, h)
    actual_output = coco_bounding_box.shift(unnormalized_bbox_shift_amount)

    assert_almost_equal(actual=list(actual_output.values), desired=list(desired))


def test_oob(coco_oob_bounding_box, image_size):
    with pytest.raises(ValueError):
        BoundingBox.from_coco(*coco_oob_bounding_box, image_size=image_size, strict=True)

    coco_box = BoundingBox.from_coco(*coco_oob_bounding_box, image_size=image_size)
    assert coco_box.is_oob is True


# Conversions


def test_to_albumentations(coco_bounding_box, albumentations_bbox):
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
