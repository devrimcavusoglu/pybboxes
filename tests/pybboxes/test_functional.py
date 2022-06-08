from pybboxes.functional import compute_area, convert_bbox
from tests.utils import assert_almost_equal


def test_convert_albumentations2voc(albumentations_bbox, voc_bbox, image_size):
    converted_box = convert_bbox(albumentations_bbox, from_type="albumentations", to_type="voc", image_size=image_size)
    assert_almost_equal(actual=list(converted_box), desired=voc_bbox)


def test_convert_coco2voc(coco_bbox, voc_bbox):
    converted_box = convert_bbox(coco_bbox, from_type="coco", to_type="voc")
    assert_almost_equal(actual=list(converted_box), desired=voc_bbox)


def test_convert_fiftyone2voc(fiftyone_bbox, voc_bbox, image_size):
    converted_box = convert_bbox(fiftyone_bbox, from_type="fiftyone", to_type="voc", image_size=image_size)
    assert_almost_equal(actual=list(converted_box), desired=voc_bbox)


def test_convert_yolo2voc(yolo_bbox, voc_bbox, image_size):
    converted_box = convert_bbox(yolo_bbox, from_type="yolo", to_type="voc", image_size=image_size)
    assert_almost_equal(actual=list(converted_box), desired=voc_bbox)


def test_convert_voc2albumentations(voc_bbox, albumentations_bbox, image_size):
    converted_box = convert_bbox(voc_bbox, from_type="voc", to_type="albumentations", image_size=image_size)
    assert_almost_equal(actual=list(converted_box), desired=albumentations_bbox)


def test_convert_voc2coco(voc_bbox, coco_bbox):
    converted_box = convert_bbox(voc_bbox, from_type="voc", to_type="coco")
    assert_almost_equal(actual=list(converted_box), desired=coco_bbox)


def test_convert_voc2fiftyone(voc_bbox, fiftyone_bbox, image_size):
    converted_box = convert_bbox(voc_bbox, from_type="voc", to_type="fiftyone", image_size=image_size)
    assert_almost_equal(actual=list(converted_box), desired=fiftyone_bbox)


def test_convert_voc2yolo(voc_bbox, yolo_bbox, image_size):
    converted_box = convert_bbox(voc_bbox, from_type="voc", to_type="yolo", image_size=image_size)
    assert_almost_equal(actual=list(converted_box), desired=yolo_bbox)


def test_area_albumentations(albumentations_bbox, bbox_area, image_size):
    area = compute_area(albumentations_bbox, bbox_type="albumentations", image_size=image_size)
    assert_almost_equal(actual=int(area), desired=bbox_area)


def test_area_coco(coco_bbox, bbox_area, image_size):
    area = compute_area(coco_bbox, bbox_type="coco", image_size=image_size)
    assert_almost_equal(actual=int(area), desired=bbox_area)


def test_area_fiftyone(fiftyone_bbox, bbox_area, image_size):
    area = compute_area(fiftyone_bbox, bbox_type="fiftyone", image_size=image_size)
    assert_almost_equal(actual=int(area), desired=bbox_area)


def test_area_voc(voc_bbox, bbox_area, image_size):
    area = compute_area(voc_bbox, bbox_type="voc", image_size=image_size)
    assert_almost_equal(actual=int(area), desired=bbox_area)


def test_area_yolo(yolo_bbox, bbox_area, image_size):
    area = compute_area(yolo_bbox, bbox_type="yolo", image_size=image_size)
    assert_almost_equal(actual=int(area), desired=bbox_area)
