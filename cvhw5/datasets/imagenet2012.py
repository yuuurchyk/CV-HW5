import os
from abc import ABC, abstractmethod
from typing import Tuple, Dict, List
import xml.etree.ElementTree as ET


import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import v2
from tqdm import tqdm


from cvhw5.utils.env import get_datasets_root
from cvhw5.augmentations import get_augmentations


class ImageNet2012(ABC, Dataset):
    def __init__(self) -> None:
        self._wnid_to_label = ImageNet2012._get_wnid_to_label()
        self._n_classes = len(self._wnid_to_label)
        self._aug = get_augmentations(augs=[])

    def set_augs(self, augs: List[str]) -> None:
        self._aug = get_augmentations(augs)

    def n_classes(self) -> None:
        return self._n_classes

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, idx) -> dict:
        pass

    def collate_fn_contrastive(self, batch):
        x = []
        y = []

        for sx, sy in batch:
            img = read_image(sx)

            assert img.ndim == 3, f'Get wrong number of dimensions: {img.ndim}'

            channels_num = img.shape[0]
            assert channels_num in (1, 3), f'Get wrong number of channels: {channels_num}'

            if channels_num == 1:
                img = img.repeat(3, 1, 1)

            img1 = self._aug(img)
            img2 = self._aug(img)

            x.append(img1)
            x.append(img2)

            y.append(sy)
            y.append(sy)

        x = torch.stack(x)
        y = torch.Tensor(y)

        return x, y

    @ staticmethod
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

    @ staticmethod
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


class ImageNetEvaluation(Dataset):
    def __init__(self, input_dataset: ImageNet2012) -> None:
        self._x = []
        self._y = []

        preprocess = v2.Compose([
            v2.Resize(224, interpolation=v2.InterpolationMode.BICUBIC),
            v2.CenterCrop(224)
        ])

        for x, y in tqdm(input_dataset):
            img = read_image(x)
            assert img.ndim == 3, f'Get wrong number of dimensions: {img.ndim}'

            channels_num = img.shape[0]
            assert channels_num in (1, 3), f'Get wrong number of channels: {channels_num}'

            if channels_num == 1:
                img = img.repeat(3, 1, 1)

            img = preprocess(img)

            self._x.append(img)
            self._y.append(y)

    def __len__(self):
        return len(self._x)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self._x[idx], self._y[idx]
