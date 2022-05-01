<h1 align="center">PyBboxes</h1>
<p align="center">
<a href="https://pypi.org/project/pybboxes"><img src="https://img.shields.io/pypi/v/pybboxes" alt="Python versions"></a>
<br>
<a href="https://pypi.org/project/pybboxes"><img src="https://img.shields.io/pypi/pyversions/pybboxes" alt="Python versions"></a>
<a href="https://github.com/devrimcavusoglu/pybboxes/actions/workflows/ci.yml"><img src="https://img.shields.io/github/workflow/status/devrimcavusoglu/pybboxes/Tests" alt="DOI"></a>
<a href="https://github.com/devrimcavusoglu/pybboxes/blob/main/LICENSE"><img src="https://img.shields.io/github/license/devrimcavusoglu/pybboxes" alt="Python versions"></a>
</p>

Light weight toolkit for bounding boxes providing conversion between bounding box types and simple computations. Supported bounding box types:

- **albumentations** : [Albumentations Format](https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/#albumentations)
- **coco** : [COCO (Common Objects in Context)](http://cocodataset.org/)
- **fiftyone** : [FiftyOne](https://github.com/voxel51/fiftyone)
- **voc** : [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/)
- **yolo** : [YOLO](https://github.com/ultralytics/yolov5)

## Installation

Through pip (recommended),

    pip install pybboxes

or build from source,

    git clone https://github.com/devrimcavusoglu/pybboxes.git
    cd pybboxes
    python setup.py install

## Conversion
You are able to convert from any bounding box type to another.

```python
import pybboxes as pbx

coco_bbox = (1,2,3,4)  # COCO Format bbox as (x-tl,y-tl,w,h)
voc_bbox = (1,2,3,4)  # Pascal VOC Format bbox as (x-tl,y-tl,x-br,y-br)
pbx.convert_bbox(coco_bbox, from_type="coco", to_type="voc")  # (1, 2, 4, 6)
pbx.convert_bbox(voc_bbox, from_type="voc", to_type="coco")  # (1, 2, 2, 2)
```

Some formats require image width and height information for scaling, e.g. YOLO bbox (resulting are round coordinates 
are rounded to 2 decimals to ease reading).

```python
import pybboxes as pbx

voc_bbox = (1,2,3,4)  # Pascal VOC Format bbox as (x-tl,y-tl,x-br,y-br)
pbx.convert_bbox(voc_bbox, from_type="voc", to_type="yolo", image_width=28, image_height=28)  # (0.07, 0.11, 0.07, 0.07)
```

## Computation
You can also make computations on supported bounding box formats.

```python
import pybboxes as pbx

coco_bbox = (1,2,3,4)  # COCO Format bbox as (x-tl,y-tl,w,h)
voc_bbox = (1,2,3,4)  # Pascal VOC Format bbox as (x-tl,y-tl,x-br,y-br)
pbx.compute_area(coco_bbox, bbox_type="coco")  # 12
pbx.compute_area(voc_bbox, bbox_type="voc")  # 4
```

### Tests

To tests simply run.

    python tests/run_tests.py

### Code Style

To check code style,

    python tests/run_code_style.py check

To format codebase,

    python tests/run_code_style.py format

## License

Licensed under the [MIT](LICENSE) License.

# TO DO

- [ ] WIDERFACE dataset will be converted to yolo and xml format. 