import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class Encoded(Dataset):
    def __init__(self, encoder: nn.Module, dataset: Dataset, device):
        self._encoder = encoder

        self._x = []
        self._y = []

        loader = DataLoader(dataset, batch_size=1024)

        with torch.no_grad():
            encoder = encoder.eval()

            for x, y in tqdm(loader):
                x = x.to(device)
                y = y.to(device)

                self._x.append(encoder(x))
                self._y.append(y)

        self._x = torch.cat(self._x)
        self._x = self._x.to(device)

        self._y = torch.cat(self._y)
        self._y = self._y.to(device)

    def __len__(self):
        return self._x.size()[0]

    def __getitem__(self, idx):
        return self._x[idx], self._y[idx]
