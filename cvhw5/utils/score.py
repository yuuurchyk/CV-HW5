from typing import Tuple
import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression


def encode(dataset: Dataset, encoder: nn.Module, batch_size: int, device: torch.device) -> Tuple[np.array, np.array]:
    dataloader = DataLoader(dataset, batch_size, pin_memory=True)

    X = []
    y = []

    with torch.no_grad():
        for bX, bY in tqdm(dataloader):
            bX = bX.float().to(device)
            bX = encoder(bX)
            bX = bX.cpu()

            X.append(bX)
            y.append(bY)

        X = torch.cat(X)
        y = torch.cat(y)

        X = X.numpy()
        y = y.numpy()

        return X, y


def train_and_score(X: np.array, y: np.array) -> float:
    logging.info(X.shape)

    model = LogisticRegression(solver='sag', penalty=None, verbose=1, max_iter=3000, n_jobs=-1)
    model.fit(X, y)

    predictions = model.predict(X)
    score = np.mean(predictions == y)

    return model, score


def score(X: np.array, y: np.array, model: LogisticRegression) -> float:
    predictions = model.predict(X)
    score = np.mean(predictions == y)
    return score
