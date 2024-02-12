import os
from typing import Tuple, Dict


from torch.utils.data import Dataset


from cvhw5.utils.env import get_datasets_root


# train subset of ImageNet 2012
class ImageNet2012Subset(Dataset):
    SUPPORTED_PERCENTS = (1, 10)

    def __init__(self, percent: int):
        assert percent in ImageNet2012Subset.SUPPORTED_PERCENTS, f'percent {percent} is not supported. Supported percents are: {ImageNet2012Subset.SUPPORTED_PERCENTS}'

        self.datasets_root = get_datasets_root()

        with open(os.path.join(self.datasets_root, f'{percent}percent.txt'), 'r') as f:
            images = f.read().splitlines()
        
        wnids = set()
        for item in images:
            wnid, _ = item.split('_')
            wnids.add(wnid)
        wnids = sorted(wnids)
        self.wnid_to_label = {}
        for i, wnid in enumerate(wnids):
            self.wnid_to_label[wnid] = i
        
        self.img_paths = []
        self.labels = []

        for item in images:
            wnid, _ = item.split('_')

            label = self.wnid_to_label[wnid]
            img_path = os.path.join(self.datasets_root, 'ILSVRC', 'Data', 'CLS-LOC', 'train', wnid, item)

            self.img_paths.append(img_path)
            self.labels.append(label)

    def wind_to_label(self) -> Dict[str, int]:
        return self.wind_to_label

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> Tuple[str, int]:
        return self.img_paths[idx], self.labels[idx]
