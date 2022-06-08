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

## Bounding Boxes

You can easily create bounding box as easy as

```python
from pybboxes import BoundingBox

my_coco_box = [98, 345, 322, 117]
coco_bbox = BoundingBox.from_coco(*my_coco_box)  # <[98 345 322 117] (322x117) | Image: (?x?)>
# or alternatively
# coco_bbox = BoundingBox.from_array(my_coco_box)
```

### Conversion

With the `BoundingBox` class the conversion is as easy as one method call.

```python
from pybboxes import BoundingBox

my_coco_box = [98, 345, 322, 117]
coco_bbox = BoundingBox.from_coco(*my_coco_box)  # <[98 345 322 117] (322x117) | Image: (?x?)>
voc_bbox = coco_bbox.to_voc()  # <[98 345 420 462] (322x117) | Image: (?x?)>
voc_bbox_values = coco_bbox.to_voc(return_values=True)  # (98, 345, 420, 462)
```

However, if you try to make conversion between two bounding boxes that require scaling/normalization it'll give an error

```python
from pybboxes import BoundingBox

my_coco_box = [98, 345, 322, 117]
coco_bbox = BoundingBox.from_coco(*my_coco_box)  # <[98 345 322 117] (322x117) | Image: (?x?)>
# yolo_bbox = coco_bbox.to_yolo()  # this will raise an exception

# You need to set image_size for coco_bbox and then you're good to go
coco_bbox.image_size = (640, 480)
yolo_bbox = coco_bbox.to_yolo()  # <[0.4047 0.8406 0.5031 0.2437] (322x117) | Image: (640x480)>
```

Image size associated with the bounding box can be given at the instantiation or while using classmethods e.g 
`from_coco()`.

```python
from pybboxes import BoundingBox

my_coco_box = [98, 345, 322, 117]
coco_bbox = BoundingBox.from_coco(*my_coco_box, image_size=(640, 480))  # <[98 345 322 117] (322x117) | Image: (640x480)>
# no longer raises exception
yolo_bbox = coco_bbox.to_yolo()  # <[0.4047 0.8406 0.5031 0.2437] (322x117) | Image: (640x480)> 
```

### Box operations

Box operations now available as of `v0.1.0`.

```python
from pybboxes import BoundingBox

my_coco_box = [98, 345, 322, 117]
my_coco_box2 = [90, 350, 310, 122]
coco_bbox = BoundingBox.from_coco(*my_coco_box, image_size=(640, 480))
coco_bbox2 = BoundingBox.from_coco(*my_coco_box2, image_size=(640, 480))

iou = coco_bbox.iou(coco_bbox2)  # 0.8117110631149508
area_union = coco_bbox + coco_bbox2  # 41670 | alternative way: coco_bbox.union(coco_bbox2)
total_area = coco_bbox.area + coco_bbox2.area  # 75494  (not union)
intersection_area = coco_bbox + coco_bbox2  # 33824 | alternative way: coco_bbox.intersection(coco_bbox2)
first_bbox_diff = coco_bbox - coco_bbox2  # 3850
second_bbox_diff = coco_bbox2 - coco_bbox  # 3996
bbox_ratio = coco_bbox / coco_bbox2 # 0.9961396086726599 (not IOU)
```

## Functional

**Note**: functional computations are moved under `pybboxes.functional` starting with the version `0.1.0`. The only 
exception is that  `convert_bbox()` which still can be used by importing `pybboxes` only (for backward compatibility).

### Conversion
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
pbx.convert_bbox(voc_bbox, from_type="voc", to_type="yolo", image_size=(28, 28))  # (0.07, 0.11, 0.07, 0.07)
```

### Computation
You can also make computations on supported bounding box formats.

```python
import pybboxes.functional as pbf

coco_bbox = (1,2,3,4)  # COCO Format bbox as (x-tl,y-tl,w,h)
voc_bbox = (1,2,3,4)  # Pascal VOC Format bbox as (x-tl,y-tl,x-br,y-br)
pbf.compute_area(coco_bbox, bbox_type="coco")  # 12
pbf.compute_area(voc_bbox, bbox_type="voc")  # 4
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