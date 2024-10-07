<h1 align="center">PyBboxes</h1>
<p align="center">
<a href="https://pypi.org/project/pybboxes"><img src="https://img.shields.io/pypi/v/pybboxes?color=blue" alt="Python versions"></a>
<a href="https://pepy.tech/project/pybboxes"><img src="https://pepy.tech/badge/pybboxes" alt="Total downloads"></a>
<a href="https://pypi.org/project/pybboxes"><img src="https://img.shields.io/pypi/dm/pybboxes?color=blue" alt="Monthly downloads"></a>
<br>
<a href="https://pypi.org/project/pybboxes"><img src="https://img.shields.io/pypi/pyversions/pybboxes" alt="Python versions"></a>
<a href="https://github.com/devrimcavusoglu/pybboxes/actions/workflows/ci.yml"><img src="https://github.com/devrimcavusoglu/pybboxes/actions/workflows/ci.yml/badge.svg" alt="Build"></a>
<a href="https://github.com/devrimcavusoglu/pybboxes/blob/main/LICENSE"><img src="https://img.shields.io/github/license/devrimcavusoglu/pybboxes" alt="Python versions"></a>
</p>

Light weight toolkit for bounding boxes providing conversion between bounding box types and simple computations. Supported bounding box types (<ins>italicized text indicates normalized values</ins>):

- **albumentations** : [Albumentations Format](https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/#albumentations)
  - **_[x-tl, y-tl, x-br, y-br]_** (Normalized VOC Format) Top-left coordinates & Bottom-right coordinates
- **coco** : [COCO (Common Objects in Context)](http://cocodataset.org/)
  - **[x-tl, y-tl, w, h]** Top-left corner & width & height
- **fiftyone** : [FiftyOne](https://github.com/voxel51/fiftyone)
  - **_[x-tl, y-tl, w, h]_** (Normalized COCO Format) Top-left coordinates & width & height
- **voc** : [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/)
  - **[x-tl, y-tl, x-br, y-br]** Top-left coordinates & Bottom-right coordinates
- **yolo** : [YOLO](https://github.com/ultralytics/yolov5)
  - **_[x-c, y-c, w, h]_** Center coordinates & width & height

**Glossary**

- **tl:** top-left
- **br:** bottom-right
- **h:** height
- **w:** width
- **c:** center


## News 🔥

* 2024/10/07 - [Annotations](#annotation-file-conversion) are supported for YOLO, COCO and VOC formats.

## Roadmap 🛣️

- [X] Annotation file support.
- [ ] (Upcoming) 3D Bounding Box support.
- [ ] (Upcoming) Polygon support.

### Important Notice
Support for Python<3.8 will be dropped starting version `0.2` though the development for Python3.6 and Python3.7 may 
continue where it will be developed under version `0.1.x` for future versions. This may introduce; however, certain 
discrepancies and/or unsupported operations in the `0.1.x` versions. To fully utilize and benefit from the entire 
package, we recommend using Python3.8 at minimum (`Python>=3.8`).

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

### Out of Bounds Boxes
Pybboxes supports OOB boxes, there exists a keyword `strict` in both Box classes (construction) and in functional 
modules. When `strict=True`, it does not allow out-of-bounds boxes to be constructed and raises an exception, while 
it does allow out-of-bounds boxes to be constructed and used when `strict=False`. Also, there is a property `is_oob` 
that indicates whether a particular bouding box is OOB or not. 

**Important** Note that, if the return value for `is_oob` is `None`, then it indicates that OOB status is unknown 
(e.g. image size required to determine, but not given). Thus, values `None` and `False` indicates different information.

```python
from pybboxes import BoundingBox

image_size = (640, 480)
my_coco_box = [98, 345, 580, 245]  # OOB box for 640x480
coco_bbox = BoundingBox.from_coco(*my_coco_box, image_size=image_size)  # Exception
# ValueError: Given bounding box values is out of bounds. To silently skip out of bounds cases pass 'strict=False'.

coco_bbox = BoundingBox.from_coco(*my_coco_box, image_size=image_size, strict=False)  # No Exception
coco_bbox.is_oob  # True
```

If you want to allow OOB, but still check OOB status, you should use `strict=False` and `is_oob` where needed.

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
intersection_area = coco_bbox * coco_bbox2  # 33824 | alternative way: coco_bbox.intersection(coco_bbox2)
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

Some formats require image width and height information for scaling, e.g. YOLO bbox (resulting coordinates 
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

## Annotation file conversion
`pybboxes` now supports the conversion of annotation file(s) across different annotation formats. (yolo, voc and coco are currently supported)

This is a 3 step process.

### 1. Instantiate the Annotations class
```python
from pybboxes.annotations import Annotations

anns = Annotations(annotation_type='yolo')
```

**Important** you have to explicitly declare `annotation_type` beforehand. post declaration, *you will be only able to load annotation in declared format* but you will be able to export to other annotation formats.

### 2. Load the annotations file
After you have instantiated the `Annotations` class declaring `annotation_type`, you can now load the annotations using appropriate method of the `Annotations` class. 

#### 2.1 Load from yolo
```python
from pybboxes.annotations import Annotations

anns = Annotations(annotation_type='yolo')

anns.load_from_yolo(labels_dir='./labels', images_dir='./images', classes_file='./classes.txt')
```

As yolo normalizes the bounding box metadata, path to corresponding  images directory must be provided (via `images_dir`) so that physical dimension of image data can be inferred.

Also, path to `classes_file` (usually classes.txt) should be provided that lists all the class labels that is used for the annotation. Without this, `pybboxes` will fail to assign appropriate class labels when converting across different annotations format.

#### 2.2 Load from voc
```python
from pybboxes.annotations import Annotations

anns = Annotations(annotation_type='voc')

anns.load_from_voc(labels_dir='./labels')
```

#### 2.3 Load from coco
```python
from pybboxes.annotations import Annotations

anns = Annotations(annotation_type='coco')

anns.load_from_coco(json_path='./validation.json')
```

### 3. Saving annotations to different format
#### 3.1 Saving annotations to yolo format
As every image data has its own corresponding annotation file in yolo format, you have to provide path to `export_dir` where all the annotation files will be written. 

```python
from pybboxes.annotations import Annotations

anns = Annotations(annotation_type='coco') # just for the demonstration purpose

anns.load_from_coco(json_path='./validation.json') # we could have loaded the annotation data from other format as well

anns.save_as_yolo(export_dir='./labels')
```
This will create all the required annotation files (in yolo format) in given directory. Additionally, it will also create `classes.txt` in the given folder which will list all the class labels used for the annotation.

#### 3.2 Saving annotations to voc format
Just like yolo format, in voc format, every image data has also its own corresponding annotation file. So, you have to provide path to `export_dir` where all the annotation files will be written.

```python
from pybboxes.annotations import Annotations

anns = Annotations(annotation_type='coco') # just for the demonstration purpose

anns.load_from_coco(json_path='./validation.json') # we could have loaded the annotation data from other format as well

anns.save_as_voc(export_dir='./labels')
```


#### 3.3 Saving annotations to coco format
To export annotations in coco format, you just have to provide name (or path) of the output file (in json format) via `export_file`.

```python
from pybboxes.annotations import Annotations

anns = Annotations(annotation_type='voc') # just for the demonstration purpose

anns.load_from_voc(labels_dir='./labels') # we could have loaded the annotation data from other format as well

anns.save_as_coco(export_file='./validation.json')
```

## Contributing

### Installation

Install the package as follows, which will set you ready for the development mode.

```shell
pip install -e .[dev]
```

### Tests

To tests simply run.

    python -m tests.run_tests

### Code Style

To check code style,

    python -m tests.run_code_style check

To format codebase,

    python -m tests.run_code_style format

## License

Licensed under the [MIT](LICENSE) License.