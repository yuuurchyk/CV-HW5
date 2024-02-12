import os
from abc import ABC, abstractmethod
from typing import Tuple, Dict
import xml.etree.ElementTree as ET


import torch
from torch.utils.data import Dataset
from torchvision.io import read_image


from cvhw5.utils.env import get_datasets_root


class ImageNet2012(ABC, Dataset):
    def __init__(self) -> None:
        self._wnid_to_label = ImageNet2012._get_wnid_to_label()
        self._n_classes = len(self._wnid_to_label)

    def n_classes(self) -> None:
        return self._n_classes

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, idx) -> dict:
        pass

    @staticmethod
    def collate_fn(batch):
        x = []
        y = []

        for sx, sy in batch:
            img = read_image(sx)

            x.append(img)
            y.append(sy)

        y = torch.Tensor(y)

        return x, y

    @staticmethod
    def _get_wnid_to_label() -> Dict[str, int]:
        base_folder = os.path.join(
            get_datasets_root(), 'ILSVRC', 'Data', 'CLS-LOC')

        wnids = set()

        for subfolder in ('train', ):
            for wnid in os.listdir(os.path.join(base_folder, subfolder)):
                wnid_path = os.path.join(base_folder, subfolder, wnid)
                assert os.path.isdir(
                    wnid_path), f'Expected {wnid_path} to be a folder'
                wnids.add(wnid)

        wnids = sorted(wnids)
        wnid_to_label = {wnid: i for i, wnid in enumerate(wnids)}

        return wnid_to_label


class ImageNet2012TrainSubset(ImageNet2012):
    SUPPORTED_PERCENTS = (1, 10)

    def __init__(self, percent: int) -> None:
        super().__init__()

        assert percent in ImageNet2012TrainSubset.SUPPORTED_PERCENTS, f'percent {percent} is not supported. Supported percents are: {ImageNet2012TrainSubset.SUPPORTED_PERCENTS}'

        datasets_root = get_datasets_root()

        with open(os.path.join(datasets_root, f'{percent}percent.txt'), 'r') as f:
            images = f.read().splitlines()

        self.img_paths = []
        self.labels = []

        for item in images:
            wnid, _ = item.split('_')

            assert wnid in self._wnid_to_label, f'unexpected wnid: {wnid}'
            label = self._wnid_to_label[wnid]
            img_path = os.path.join(
                datasets_root, 'ILSVRC', 'Data', 'CLS-LOC', 'train', wnid, item)

            self.img_paths.append(img_path)
            self.labels.append(label)

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> Tuple[str, int]:
        return self.img_paths[idx], self.labels[idx]


class ImageNet2012Validation(ImageNet2012):
    def __init__(self) -> None:
        super().__init__()

        self.img_paths = []
        self.labels = []

        datasets_root = get_datasets_root()
        annotations_folder = os.path.join(datasets_root, 'ILSVRC', 'Annotations', 'CLS-LOC', 'val')
        images_folder = os.path.join(datasets_root, 'ILSVRC', 'Data', 'CLS-LOC', 'val')

        for annotation_file in os.listdir(annotations_folder):
            assert os.path.isfile(os.path.join(annotations_folder, annotation_file))
            assert annotation_file.lower().endswith('.xml')

            wnid = ImageNet2012Validation._parse_xml(os.path.join(annotations_folder, annotation_file))
            assert wnid in self._wnid_to_label, f'unexpected wnid: {wnid}'
            label = self._wnid_to_label[wnid]

            filename, _ = os.path.splitext(annotation_file)

            image_file = os.path.join(images_folder, f'{filename}.JPEG')
            assert os.path.isfile(image_file), f'Could not find image file: {image_file}'

            self.img_paths.append(image_file)
            self.labels.append(label)

    @staticmethod
    def _parse_xml(xml_filepath: str) -> str:
        with open(xml_filepath, 'r') as f:
            content = f.read()

        root = ET.fromstring(content)

        objects = root.findall('object')
        assert len(objects) > 0, f'Xml file {xml_filepath}: no objects found!'

        present_wnids = set()

        for object in objects:
            wnid = object.find('name').text
            present_wnids.add(wnid)

        assert len(present_wnids) == 1, f'Found multiple wnids in xml file: {xml_filepath}'

        return list(present_wnids)[0]

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> Tuple[str, int]:
        return self.img_paths[idx], self.labels[idx]
