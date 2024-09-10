from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Union
import os

import numpy as np
from pycocotools.coco import COCO

from pybboxes.boxes import BoundingBox


@dataclass
class Annotation:
    # https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch
    boxes: List[BoundingBox] # in the same annotation format/type
    label_id: int
    label_name: str = None
    annotation_id: int = None
    annotation_type: str = None
    associated_image_name:str = None
    segmentations: List[int] = None
    image_width: int = None
    image_height: int = None

class Annotations:
    def __init__(self, annotation_type:str):
        valid_types = ('yolo', 'coco', 'voc', 'albumentations', 'fiftyone')
        if annotation_type not in valid_types:
            raise ValueError(f"Annotation type should be one of: {valid_types}")
        
        self._annotation_type = annotation_type
        self._class_name: List[str] = []
        self._objects: List[Annotation] = []

    @property
    def names_mapping(self):
        return {name: id_ for id_, name in enumerate(self._class_names)}

    def __getitem__(self, subscript: Union[int, List[int], slice]) -> Union[Annotation, List[Annotation]]:
        if isinstance(subscript, list):
            return [self[i] for i in subscript]
        else:
            return self._objects[subscript]

    def label2id(self, name: str):
        return self.names_mapping[name]

    def id2label(self, label_id: int):
        return self._class_names[label_id]
    
    def load_from_coco(self, json_path:str):
        """
        initializes Annotations from coco json file

        Parameters
        ----------
        json_path : str
            provide path to coco annotation  file in json format
        """
        if self._annotation_type != 'coco':
            raise ValueError(f'this instance of Annotations can only process {self._annotation_type} annotation file(s)')

        if not os.path.exists(json_path):
            raise FileNotFoundError(f"{json_path} doesn't exists")
        
        coco = COCO(json_path)

        categories = coco.loadCats(coco.getCatIds())
        self._class_names = [category['name'] for category in categories] # we just need the names

        for ann_id in coco.getAnnIds():
            ann = coco.loadAnns(ann_id)[0]
            img = coco.loadImgs(ann['image_id'])[0]

            bbox = BoundingBox.from_coco(*ann['bbox'], image_size=(img['width'], img['height']))

            annotation = Annotation(
                boxes=[bbox],
                label_id=ann['category_id'],
                label_name=self.id2label(ann['category_id']),
                annotation_type='coco',
                associated_image_name=img['file_name'],
                annotation_id=ann['id'],
                image_width=img['width'],
                image_height=img['height']
            )

            self._objects.append(annotation)

    def persist_as_yolo(self, export_dir: str):
        os.makedirs(export_dir, exist_ok=True)
        for annotation in self._objects:

            filename = f'{os.path.splitext(annotation.associated_image_name)[0]}.txt'
            filepath = os.path.join(export_dir, filename)

            with open(filepath, mode='a') as f:
                for box in annotation.boxes:
                    yolo_box = box.to_yolo().raw_values
                    yolo_box = [f'{x:.4f}' for x in yolo_box]
                    yolo_box.insert(0, str(annotation.label_id)) # append class/label id at the beginning
                    f.write(f"{' '.join(yolo_box)}\n")

if __name__ == "__main__":
    coco = Annotations(annotation_type='coco')
    coco.load_from_coco(json_path='Images/annotations_coco.json')
    coco.persist_as_yolo(export_dir='yolo_test')