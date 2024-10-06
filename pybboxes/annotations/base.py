import json
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import List

from pycocotools.coco import COCO

from pybboxes.boxes import BoundingBox
from pybboxes.utils.io import get_image_size


@dataclass
class Annotation:
    # This is the format in which 'Annotations' class store the bounding box details internally
    # Single instance can store the info of only one bounding box
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
    def __init__(self, annotation_type: str):
        """Initializes Annotations of defined format

        Parameters
        ----------
        annotation_type : str
            should be within (yolo, coco, voc, albumentations and fiftyone)

        Raises
        ------
        ValueError
            if annotation_type is not of supported type
        """
        valid_types = ("yolo", "coco", "voc", "albumentations", "fiftyone")
        if annotation_type not in valid_types:
            raise ValueError(f"Annotation type should be one of: {valid_types}")

        self._annotation_type = annotation_type
        self._class_names: List[str] = []
        self._objects: dict[str, List[Annotation]] = dict()

    @property
    def names_mapping(self):
        """lists all the classes in dictionary format

        Returns
        -------
        dict
            in {class_name:class_id, ...} format
        """
        return {name: id_ for id_, name in enumerate(self._class_names)}

    # def __getitem__(self, subscript: Union[int, List[int], slice]) -> Union[Annotation, List[Annotation]]:
    #     if isinstance(subscript, list):
    #         return [self[i] for i in subscript]
    #     else:
    #         return self._objects[subscript]

    def label2id(self, name: str):
        """returns class id for the given class name

        Parameters
        ----------
        name : str

        Returns
        -------
        int
        """
        return self.names_mapping[name]

    def id2label(self, label_id: int):
        """returns class name for the given class label

        Parameters
        ----------
        label_id : int

        Returns
        -------
        str
        """
        return self._class_names[label_id]

    def load_from_albumentations(self):
        raise NotImplementedError

    def load_from_fiftyone(self):
        raise NotImplementedError

    def load_from_voc(self, labels_dir: str):
        """
        initializes Annotations from xml annotations in pascal voc format

        Parameters
        ----------
        labels_dir : str
            provide path to directory that houses xml annotations in pascal voc format
        """
        if self._annotation_type != "voc":
            raise TypeError(f"this instance of Annotations can only process {self._annotation_type} annotation file(s)")

        if not os.path.exists(labels_dir):
            raise FileNotFoundError(f"{labels_dir} doesn't exists")

        for filename in os.listdir(labels_dir):
            if filename.endswith(".xml"):
                tree = ET.parse(os.path.join(labels_dir, filename))
                root = tree.getroot()

                image_name = root.find("filename").text
                size = root.find("size")
                img_w = int(size.find("width").text)
                img_h = int(size.find("height").text)

                for obj in root.findall("object"):
                    label_name = obj.find("name").text
                    if label_name not in self._class_names:
                        self._class_names.append(label_name)
                    label_id = self.label2id(label_name)

                    bbox = obj.find("bndbox")
                    xmin = float(bbox.find("xmin").text)
                    ymin = float(bbox.find("ymin").text)
                    xmax = float(bbox.find("xmax").text)
                    ymax = float(bbox.find("ymax").text)

                    bbox = BoundingBox.from_voc(xmin, ymin, xmax, ymax, image_size=(img_w, img_h))

                    annotatation = Annotation(
                        box=bbox,
                        label_id=label_id,
                        label_name=label_name,
                        annotation_type="voc",
                        image_width=img_w,
                        image_height=img_h,
                    )

                    if image_name in self._objects:
                        self._objects[image_name].append(annotatation)
                    else:
                        self._objects[image_name] = [annotatation]

    def load_from_coco(self, json_path: str):
        """
        initializes Annotations from coco annotation file (json files)

        Parameters
        ----------
        json_path : str
            provide path to coco annotation  file in json format
        """
        if self._annotation_type != "coco":
            raise TypeError(f"this instance of Annotations can only process {self._annotation_type} annotation file(s)")

        if not os.path.exists(json_path):
            raise FileNotFoundError(f"{json_path} doesn't exists")

        coco = COCO(json_path)

        categories = coco.loadCats(coco.getCatIds())
        self._class_names = [category["name"] for category in categories]  # we just need the names

        for ann_id in coco.getAnnIds():
            ann = coco.loadAnns(ann_id)[0]
            img = coco.loadImgs(ann["image_id"])[0]
            associated_img_filename = img["file_name"]

            bbox = BoundingBox.from_coco(*ann["bbox"], image_size=(img["width"], img["height"]))

            annotation = Annotation(
                box=bbox,
                label_id=ann["category_id"],
                label_name=self.id2label(ann["category_id"]),
                annotation_type="coco",
                annotation_id=ann["id"],
                image_width=img["width"],
                image_height=img["height"],
            )

            # if the entry for this file already exists
            if associated_img_filename in self._objects:
                self._objects[associated_img_filename].append(annotation)
            else:
                self._objects[associated_img_filename] = [annotation]

    def load_from_yolo(self, labels_dir: str, images_dir: str, classes_file: str):
        """load annoations in yolo format

        Parameters
        ----------
        labels_dir : str
            immediate parent directory that houses all the image annoatations
        images_dir : str
            immediate parent directory that houses all the images (we need corresponding images to labels to extract image dimensions)
        classes_file : str
            path to classes.txt that lists all the class labels used in the annotation
        """

        if self._annotation_type != "yolo":
            raise TypeError(f"this instance of Annotations can only process {self._annotation_type} annotation file(s)")

        if not os.path.exists(classes_file):
            raise FileNotFoundError(f"{classes_file} doesn't exist")

        if not os.path.exists(labels_dir):
            raise NotADirectoryError(f"{labels_dir} is not a valid directory")

        if not os.path.exists(images_dir):
            raise NotADirectoryError(f"{images_dir} is not a valid directory")

        with open(classes_file, "r") as f:
            self._class_names = [line.strip() for line in f.readlines()]

        for filename in os.listdir(labels_dir):
            if filename.endswith(".txt"):
                if "classes" in filename:
                    continue  # if this is classes label, we have to skip as it donot contains bounding boxes data

                image_name = filename.replace(".txt", ".jpg")  # we are assuming jpg extension
                if not os.path.exists(os.path.join(images_dir, image_name)):
                    image_name = filename.replace(".jpg", ".jpeg")  # see if image with jpeg extension exits
                    if not os.path.exists(image_name):
                        raise FileNotFoundError(f"{image_name} not found in images directory")

                image_size = get_image_size(os.path.join(images_dir, image_name))  # we need for yolo format

                with open(os.path.join(labels_dir, filename), "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        label_id = int(parts[0])  # extract the class/label id
                        x_c, y_c, w, h = map(float, parts[1:5])

                        bbox = BoundingBox.from_yolo(x_c, y_c, w, h, image_size=image_size)

                        annotation = Annotation(
                            box=bbox,
                            label_id=label_id,
                            label_name=self.id2label(label_id),
                            annotation_type="yolo",
                            image_width=image_size[0],
                            image_height=image_size[1],
                        )

                        if image_name not in self._objects.keys():
                            self._objects[image_name] = [annotation]
                        else:
                            self._objects[image_name].append(annotation)

    def save_as_yolo(self, export_dir: str):
        """writes loaded annotations in yolo format

        Parameters
        ----------
        export_dir : str
            path to directory where all the annotation files should be written

        this will write annotation files for all the corresponding images and also 'classes.txt' that defines all the class
        used for the annotation
        """
        os.makedirs(export_dir, exist_ok=True)

        # write class file
        with open(os.path.join(export_dir, "classes.txt"), "w") as f:
            for cls in self._class_names:
                f.write(f"{cls}\n")

        for image_name in self._objects.keys():

            filename = f"{os.path.splitext(image_name)[0]}.txt"
            filepath = os.path.join(export_dir, filename)

            with open(filepath, mode="a") as f:
                for annotation in self._objects[image_name]:
                    yolo_box = annotation.box.to_yolo().raw_values
                    yolo_box = [f"{x:.4f}" for x in yolo_box]
                    yolo_box.insert(0, str(annotation.label_id))  # append class/label id at the beginning
                    f.write(f"{' '.join(yolo_box)}\n")

    def save_as_voc(self, export_dir: str, n_channels: int = 3):
        """writes loaded annotations in voc format

        Parameters
        ----------
        export_dir : str
            path to directory where all the annotation files should be written
        """
        os.makedirs(export_dir, exist_ok=True)
        for image_name in self._objects.keys():
            filename = os.path.splitext(image_name)[0] + ".xml"
            filepath = os.path.join(export_dir, filename)

            root = ET.Element("annotation")
            ET.SubElement(root, "filename").text = image_name
            size = ET.SubElement(root, "size")

            if len(self._objects[image_name]) == 0:
                raise ValueError(f"no associated annotations for {image_name}")

            # get the first sample from list because it contains image dimensions
            sample_annotation = self._objects[image_name][0]
            ET.SubElement(size, "width").text = str(sample_annotation.image_width)
            ET.SubElement(size, "height").text = str(sample_annotation.image_height)
            ET.SubElement(size, "depth").text = str(n_channels)
            del sample_annotation  # after we have extracted image width and height, we donot need it anymore

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

    def save_as_coco(self, export_file: str):
        """writes loaded annotation in coco format (json format)

        Parameters
        ----------
        export_file : str
            name (or path) for the annotation file
        """
        coco_data = {"images": [], "categories": [], "annotations": []}

        # embed categorical information
        for i, name in enumerate(self._class_names):
            coco_data["categories"].append({"id": i, "name": name, "supercategory": "none"})

        image_id = 0
        ann_id = 0
        for image_name in self._objects.keys():
            for annotation in self._objects[image_name]:
                # embed image metadata
                coco_data["images"].append(
                    {
                        "id": image_id,
                        "file_name": image_name,
                        "width": annotation.image_width,
                        "height": annotation.image_height,
                    }
                )

                # embed annotation metadata
                coco_box = annotation.box.to_coco().raw_values
                coco_data["annotations"].append(
                    {
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": annotation.label_id,
                        "bbox": coco_box,
                        "area": coco_box[2] * coco_box[3],
                        "iscrowd": 0,
                    }
                )

                ann_id += 1
            image_id += 1

        with open(export_file, "w", encoding="utf-8") as f:
            json.dump(coco_data, f)

    def save_as_albumentations(self):
        raise NotImplementedError

    def save_as_fiftyone(self):
        raise NotImplementedError
