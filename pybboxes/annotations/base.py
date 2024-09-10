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

    def __getitem__(
        self, subscript: Union[int, List[int], slice]
    ) -> Union[Annotation, List[Annotation]]:
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
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"{json_path} doesn't exists")
        
        coco = COCO(json_path)

        def load_image_metadata(img_id: int):
            return coco.loadImgs(img_id)[0] 

        def augment_annotation_with_image_metadata(annotation: dict):
            img_id = annotation['image_id']
            annotation['image'] = load_image_metadata(img_id)
            return annotation

        categories = coco.loadCats(coco.getCatIds())
        categories = [category['name'] for category in categories] # we just need the names
        self._class_names = categories

        annotations = coco.loadAnns(coco.getAnnIds())
        annotations = map(augment_annotation_with_image_metadata, annotations) # we will need image metadata for image file name and dimensions

        for annotation in annotations:
            ann = Annotation(
                boxes=[BoundingBox.from_coco(*annotation['bbox'], image_size=(annotation['image']['width'], annotation['image']['height']))],
                label_id=annotation['category_id'],
                label_name=self.id2label(annotation['category_id']),
                annotation_type=self._annotation_file_type,
                associated_image_name=annotation['image']['file_name'],
                annotation_id=annotation['id']
            )
            
            self._objects.append(ann)
    
    def persist_as_yolo(self, export_dir: str):
        os.makedirs(export_dir, exist_ok=True)
        for annotation in self._objects:

            filename = os.path.splitext(annotation.associated_image_name)[0]
            filepath = os.path.join(export_dir, f'{filename}.txt')

            writing_mode = 'a' if os.path.exists(filepath) else 'w' # some file may contain multiple bounding boxes, in such case, use append prevent over write
            with open(filepath, mode=writing_mode) as f:
                bboxes = annotation.boxes[0]
                bboxes = bboxes.to_yolo() # convert to yolo format
                bboxes = bboxes.raw_values
                bboxes = [f'{x:.4f}' for x in bboxes] # convert from float to string and round to 4 decimal places
                bboxes.insert(0, str(annotation.label_id)) # append the class label at the beginning
                bboxes = ' '.join(bboxes) # this represents annotation for a single line
                f.write(f'{bboxes}\n')

if __name__ == "__main__":
    coco = Annotations(annotation_file_type='coco')
    coco.load_from_coco(json_path='Images/annotations_coco.json')