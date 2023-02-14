from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Union

import numpy as np

from pybboxes.boxes import BoundingBox


@dataclass
class Annotation:
    # https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch
    boxes: List[BoundingBox]
    label_id: int
    label_name: str = None
    annotation_id = None
    segmentations: List[int] = None


class Annotations(ABC):
    def __init__(self, class_names: List[str]):
        self.class_names = class_names
        self.objects = []

    @property
    def names_mapping(self):
        return {name: id_ for id_, name in enumerate(self.class_names)}

    @abstractmethod
    def add(self, boxes: Union[List, np.ndarray, Tuple]) -> None:
        """
        Adds a single or multiple bounding boxes for objects.

        Args:
            boxes:

        Returns:

        """
        pass

    def __getitem__(
        self, subscript: Union[int, List[int], slice]
    ) -> Union[Annotation, List[Annotation]]:
        if isinstance(subscript, list):
            return [self[i] for i in subscript]
        else:
            return self.objects[subscript]

    def label2id(self, name: str):
        return self.names_mapping[name]

    def id2label(self, label_id: int):
        return self.class_names[label_id]
