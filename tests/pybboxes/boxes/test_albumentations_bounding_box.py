import numpy as np
import pytest

from pybboxes import AlbumentationsBoundingBox, BoundingBox
from tests.utils import assert_almost_equal


@pytest.fixture(scope="function")
def albumentations_bounding_box(albumentations_bbox, image_size):
    return BoundingBox.from_albumentations(*albumentations_bbox, image_size=image_size)


@pytest.fixture(scope="module")
def albumentations_oob_bounding_box():
    return [0.15625, 0.21875, 0.875, 1.1041666666666667]


@pytest.fixture(scope="module")
def albumentations_bounding_box2(albumentations_bbox, image_size):
    np.random.seed(42)
    albumentations_bbox2 = albumentations_bbox + np.random.uniform(-0.05, 0.05, size=4)
    return BoundingBox.from_albumentations(*albumentations_bbox2, image_size=image_size)


@pytest.fixture()
def scaled_albumentations_box():
    return 0.22680595035775913, 0.7544463610428895, 0.5825690496422409, 0.9268036389571105


@pytest.fixture()
def clamped_albumentations_box():
    return 0.22680595035775913, 0.7544463610428895, 0.5825690496422409, 0.9268036389571105


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


def test_from_array(albumentations_bbox, image_size):
    with pytest.warns(FutureWarning):
        alb_box = AlbumentationsBoundingBox.from_array(albumentations_bbox, image_size=image_size)

    assert alb_box.is_oob is False


@pytest.mark.parametrize(
    "box_values,expected_out",
    [
        ((0.15625, 0.21875, 0.875, 1.1041666666666667), (0.15625, 0.21875, 0.875, 1)),
        ((-0.05, -0.05, 0.5, 0.5), (0.0, 0.0, 0.5, 0.5)),
        ((0.3, 0.4, 1.04, 0.7), (0.3, 0.4, 1.0, 0.7)),
    ],
)
def test_clamp(box_values, expected_out, image_size):
    alb_box = AlbumentationsBoundingBox(*box_values, image_size=image_size)
    alb_box.clamp()

    assert_almost_equal(actual=alb_box.values, desired=expected_out, ignore_numeric_type_changes=True)


def test_scale(albumentations_bounding_box, scaled_albumentations_box, scale_factor):
    x_tl, y_tl, x_br, y_br = albumentations_bounding_box.values
    image_width, image_height = albumentations_bounding_box.image_size
    w, h = (x_br - x_tl) * image_width, (y_br - y_tl) * image_height

    albumentations_bounding_box.scale(scale_factor)

    assert_almost_equal(
        actual=albumentations_bounding_box.values, desired=scaled_albumentations_box, ignore_numeric_type_changes=True
    )

    actual_area = albumentations_bounding_box.area
    desired_area = w * h * scale_factor
    assert actual_area - desired_area < 10**2


def test_shift(albumentations_bounding_box, normalized_bbox_shift_amount):
    x_tl, y_tl, x_br, y_br = albumentations_bounding_box.values
    desired = (
        x_tl + normalized_bbox_shift_amount[0],
        y_tl + normalized_bbox_shift_amount[1],
        x_br + normalized_bbox_shift_amount[0],
        y_br + normalized_bbox_shift_amount[1],
    )
    actual_output = albumentations_bounding_box.shift(normalized_bbox_shift_amount)

    assert_almost_equal(actual=actual_output.values, desired=desired, decimal=2)


def test_oob(albumentations_oob_bounding_box, image_size):
    with pytest.raises(ValueError):
        BoundingBox.from_albumentations(*albumentations_oob_bounding_box, image_size=image_size, strict=True)

    alb_box = BoundingBox.from_albumentations(*albumentations_oob_bounding_box, image_size=image_size)
    assert alb_box.is_oob is True


# Conversions


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
