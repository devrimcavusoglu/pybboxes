from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Union, Dict
import xml.etree.ElementTree as ET
import json
import os

import numpy as np
from pycocotools.coco import COCO

from pybboxes.boxes import BoundingBox


@dataclass
class Annotation:
    # https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch
    box: BoundingBox
    label_id: int
    label_name: str = None
    annotation_id: int = None
    annotation_type: str = None
    segmentations: List[int] = None
    image_width: int = None
    image_height: int = None

class Annotations:
    def __init__(self, annotation_type:str):
        valid_types = ('yolo', 'coco', 'voc', 'albumentations', 'fiftyone')
        if annotation_type not in valid_types:
            raise ValueError(f"Annotation type should be one of: {valid_types}")
        
        self._annotation_type = annotation_type
        self._class_names: List[str] = []
        self._objects: dict[str, List[Annotation]] = dict()

    @property
    def names_mapping(self):
        return {name: id_ for id_, name in enumerate(self._class_names)}

    # def __getitem__(self, subscript: Union[int, List[int], slice]) -> Union[Annotation, List[Annotation]]:
    #     if isinstance(subscript, list):
    #         return [self[i] for i in subscript]
    #     else:
    #         return self._objects[subscript]

    def label2id(self, name: str):
        return self.names_mapping[name]

    def id2label(self, label_id: int):
        return self._class_names[label_id]
    
    def load_from_voc(self, annotations_directory: str):
        """
        initializes Annotations from xml annotations in pascal voc format

        Parameters
        ----------
        annotations_directory : str
            provide path to directory that houses xml annotations in pascal voc format
        """
        if self._annotation_type != 'voc':
            raise ValueError(f'this instance of Annotations can only process {self._annotation_type} annotation file(s)')

        if not os.path.exists(annotations_directory):
            raise FileNotFoundError(f"{annotations_directory} doesn't exists")
        
        for filename in os.listdir(annotations_directory):
            if filename.endswith('.xml'):
                tree = ET.parse(os.path.join(annotations_directory, filename))
                root = tree.getroot()

                image_name = root.find('filename').text
                size = root.find('size')
                img_w = int(size.find('width').text)
                img_h = int(size.find('height').text)

                for obj in root.findall('object'):
                    label_name = obj.find('name').text
                    if label_name not in self._class_names:
                        self._class_names.append(label_name)
                    label_id = self.label2id(label_name)

                    bbox = obj.find('bndbox')
                    xmin = float(bbox.find('xmin').text)
                    ymin = float(bbox.find('ymin').text)
                    xmax = float(bbox.find('xmax').text)
                    ymax = float(bbox.find('ymax').text)

                    bbox = BoundingBox.from_voc(xmin, ymin, xmax, ymax, image_size=(img_w, img_h))

                    annotatation = Annotation(
                        box=bbox,
                        label_id=label_id,
                        label_name=label_name,
                        annotation_type='voc',
                        image_width=img_w,
                        image_height=img_h
                    )

                    if image_name in self._objects:
                        self._objects[image_name].append(annotatation)
                    else:
                        self._objects[image_name] = [annotatation]


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
            associated_img_filename = img['file_name']

            bbox = BoundingBox.from_coco(*ann['bbox'], image_size=(img['width'], img['height']))

            annotation = Annotation(
                box=bbox,
                label_id=ann['category_id'],
                label_name=self.id2label(ann['category_id']),
                annotation_type='coco',
                annotation_id=ann['id'],
                image_width=img['width'],
                image_height=img['height']
            )

            # if the entry for this file already exists
            if associated_img_filename in self._objects:
                self._objects[associated_img_filename].append(annotation)
            else:
                self._objects[associated_img_filename] = [annotation]

    def persist_as_yolo(self, export_dir: str):
        os.makedirs(export_dir, exist_ok=True)
        for image_name in self._objects.keys():

            filename = f'{os.path.splitext(image_name)[0]}.txt'
            filepath = os.path.join(export_dir, filename)

            with open(filepath, mode='a') as f:
                for annotation in self._objects[image_name]:
                    yolo_box = annotation.box.to_yolo().raw_values
                    yolo_box = [f'{x:.4f}' for x in yolo_box]
                    yolo_box.insert(0, str(annotation.label_id)) # append class/label id at the beginning
                    f.write(f"{' '.join(yolo_box)}\n")

    def persist_as_voc(self, export_dir: str, n_channels: int=3):
        os.makedirs(export_dir, exist_ok=True)
        for image_name in self._objects.keys():
            filename = os.path.splitext(image_name)[0] + '.xml'
            filepath = os.path.join(export_dir, filename)

            root = ET.Element("annotation")
            ET.SubElement(root, "filename").text = image_name
            size = ET.SubElement(root, "size")

            if len(self._objects[image_name]) == 0:
                raise ValueError(f'no associated annotations for {image_name}')
            
            # get the first sample from list because it contains image dimensions
            sample_annotation = self._objects[image_name][0]
            ET.SubElement(size, "width").text = str(sample_annotation.image_width)
            ET.SubElement(size, "height").text = str(sample_annotation.image_height)
            ET.SubElement(size, "depth").text = str(n_channels)
            del sample_annotation # after we have extracted image width and height, we donot need it anymore

            for annotation in self._objects[image_name]:
                obj = ET.SubElement(root, "object")
                ET.SubElement(obj, "name").text = annotation.label_name
                ET.SubElement(obj, "pose").text = "Unspecified"
                ET.SubElement(obj, "truncated").text = "0"
                ET.SubElement(obj, "difficult").text = "0"

                bbox = ET.SubElement(obj, "bndbox")
                voc_box = annotation.box.to_voc().raw_values
                ET.SubElement(bbox, "xmin").text = str(int(voc_box[0]))
                ET.SubElement(bbox, "ymin").text = str(int(voc_box[1]))
                ET.SubElement(bbox, "xmax").text = str(int(voc_box[2]))
                ET.SubElement(bbox, "ymax").text = str(int(voc_box[3]))

            tree = ET.ElementTree(root)
            tree.write(filepath)

    def persist_as_coco(self, export_file: str):
        coco_data = {
            "images": [],
            "categories": [],
            "annotations": []
        }

        # embed categorical information
        for i, name in enumerate(self._class_names):
            coco_data['categories'].append({
                "id": i,
                "name": name,
                "supercategory": "none"
            })

        image_id = 0
        ann_id = 0
        for image_name in self._objects.keys():
            for annotation in self._objects[image_name]:
                # embed image metadata
                coco_data['images'].append(
                    {
                        "id": image_id,
                        "file_name": image_name,
                        "width": annotation.image_width,
                        "height": annotation.image_height
                    }
                )

                # embed annotation metadata
                coco_box = annotation.box.to_coco().raw_values
                coco_data['annotations'].append(
                    {
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": annotation.label_id,
                        "bbox": coco_box,
                        "area": coco_box[2] * coco_box[3],
                        "iscrowd": 0
                    }
                )

                ann_id += 1
            image_id += 1

        with open(export_file, 'w', encoding='utf-8') as f:
            json.dump(coco_data, f)
            
if __name__ == "__main__":
    coco = Annotations(annotation_type='voc')
    coco.load_from_voc('voc_test')
    # coco.load_from_coco(json_path='Images/annotations_coco.json')
    # coco.persist_as_yolo(export_dir='yolo_test')
    coco.persist_as_coco(export_file='coco_test.json')
    # coco.persist_as_voc(export_dir='voc_test')