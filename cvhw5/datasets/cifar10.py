import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms.v2 as v2

from cvhw5.utils.env import get_datasets_root


class Cifar10(Dataset):
    def __init__(self, train: bool, transform, device: torch.device):
        dataset = torchvision.datasets.CIFAR10(root=get_datasets_root(), train=train, download=True, transform=v2.Compose([
            v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]))

        self._transform = transform

        self._x = []
        self._y = []

        for i in range(len(dataset)):
            sx, sy = dataset[i]

            self._x.append(sx)
            self._y.append(sy)

        self._x = torch.stack(self._x).to(device)
        self._y = torch.tensor(self._y).to(device)

    def __len__(self):
        return len(self._x)

    def __getitem__(self, idx):
        return self._transform(self._x[idx]), self._y[idx]


class Cifar10Contrastive(Dataset):
    def __init__(self, train: bool, transform, device: torch.device):
        dataset = torchvision.datasets.CIFAR10(root=get_datasets_root(), train=train, download=True, transform=v2.Compose([
            v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]))

        self._transform = transform

        self._x = []
        self._y = []

        for i in range(len(dataset)):
            sx, sy = dataset[i]

            self._x.append(sx)
            self._y.append(sy)

        self._x = torch.stack(self._x).to(device)
        self._y = torch.tensor(self._y).to(device)

    def collate_fn(self, batch):
        x = []
        y = []

        for img, sy in batch:
            img1 = self._transform(img)
            img2 = self._transform(img)

            x.append(img1)
            x.append(img2)

            y.append(sy)
            y.append(sy)

        x = torch.stack(x)
        y = torch.Tensor(y)

        return x, y

    def __len__(self):
        return len(self._x)

    def __getitem__(self, idx):
        return self._x[idx], self._y[idx]
