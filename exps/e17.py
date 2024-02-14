import os
import logging

import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


from cvhw5.loss import get_contrastive_loss
from cvhw5.utils.logger import set_up_logger, set_up_logging_to_file
from cvhw5.utils.exp import get_exp_folder
from cvhw5.datasets.cifar10 import Cifar10Contrastive
from cvhw5.augmentations import get_augmentations


# contrastive loss of CIFAR10, bigger batch, more augmentations, efficientnetB6
EXP_NAME = "exp17"
EPOCHS_NUM = 2500
BATCH_SIZE = 1024
EMBEDDING_SIZE = 128
LR = 1e-4
LR_DECAY = 0.85
LR_DECAY_EPOCHS = 100
PROJECTION_HIDDEN_DIM = 1024
EVALUATION_MODEL_EPOCHS = 50
SAVE_MODEL_EPOCHS = 100
AUGS = ['colordistortion', 'gaussianblur', 'horizontalflip']


def calc_loss(model, dataloader, device):
    loss_sum = 0.0
    loss_n = 0

    for batch_x, _ in tqdm(dataloader):
        batch_x = batch_x.to(device)

        z = model(batch_x)
        loss = get_contrastive_loss(z)

        loss_sum += loss.item()
        loss_n += 1

    return loss_sum / loss_n


def main():
    set_up_logger()

    exp_folder = get_exp_folder(EXP_NAME)

    set_up_logging_to_file(os.path.join(exp_folder, 'exp.log'))

    torch.manual_seed(47)
    device = torch.device('cuda')

    encoder = torchvision.models.efficientnet_b6()
    projection = nn.Sequential(
        nn.Linear(1000, PROJECTION_HIDDEN_DIM),
        nn.ReLU(),
        nn.Linear(PROJECTION_HIDDEN_DIM, EMBEDDING_SIZE)
    )
    model = nn.Sequential(encoder, projection)
    model = model.to(device)

    writer = SummaryWriter(exp_folder)

    logging.info('Loading datasets')

    train_dataset = Cifar10Contrastive(True, get_augmentations(
        AUGS, random_resized_crop_size=32), device=torch.device('cpu'))
    validation_dataset = Cifar10Contrastive(False, get_augmentations(
        [], random_resized_crop_size=32), device=torch.device('cpu'))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              collate_fn=train_dataset.collate_fn, shuffle=True, drop_last=True, pin_memory=True, num_workers=32)
    validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE,
                                   collate_fn=train_dataset.collate_fn, shuffle=False, pin_memory=True, num_workers=32)

    optimizer = Adam(model.parameters(), lr=LR)
    scheduler = StepLR(optimizer, step_size=LR_DECAY_EPOCHS, gamma=LR_DECAY)

    tick = 0

    for epoch in range(1, EPOCHS_NUM + 1):
        logging.info(f'Epoch {epoch}')

        writer.add_scalar('LR', scheduler.get_last_lr()[0], epoch)

        model = model.train()

        for batch_X, _ in tqdm(train_loader):
            optimizer.zero_grad()

            batch_X = batch_X.to(device)

            z = model(batch_X)

            loss = get_contrastive_loss(z)

            writer.add_scalar('train_loss', loss.item(), tick)

            loss.backward()
            optimizer.step()

            tick += 1

        if epoch % EVALUATION_MODEL_EPOCHS == 0:
            model = model.eval()

            with torch.no_grad():
                logging.info(f'Evaluating train')
                ltrain = calc_loss(model, train_loader, device)

                logging.info(f'Evaluating validation')
                lvalidation = calc_loss(model, validation_loader, device)

                writer.add_scalars('avg_loss', {
                    'train': ltrain,
                    'validation': lvalidation
                }, epoch)

        if epoch % SAVE_MODEL_EPOCHS == 0:
            logging.info('Saving model')

            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }, os.path.join(exp_folder, f'{epoch}.pth'))

        scheduler.step()


if __name__ == '__main__':
    main()
