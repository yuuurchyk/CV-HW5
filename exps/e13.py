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


from cvhw5.utils.logger import set_up_logger, set_up_logging_to_file
from cvhw5.utils.exp import get_exp_folder
from cvhw5.augmentations import get_augmentations
from cvhw5.datasets.cifar10 import Cifar10


# classification on CIFAR10
EXP_NAME = "exp13"
EPOCHS_NUM = 500
BATCH_SIZE = 2048
LR = 1e-3
LR_DECAY = 0.85
LR_DECAY_EPOCHS = 20
EVALUATION_MODEL_EPOCHS = 10
SAVE_MODEL_EPOCHS = 50
AUGS = ['colordistortion', 'gaussianblur', 'horizontalflip']


def calc_accuracy(model, dataloader, device):
    correct = 0
    total = 0

    for batch_X, batch_y in tqdm(dataloader):
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        logits = model(batch_X)
        _, predicted = torch.max(logits, 1)

        total += batch_y.size()[0]
        correct += (predicted == batch_y).sum().item()

    accuracy = correct / total

    return accuracy


def main():
    set_up_logger()

    exp_folder = get_exp_folder(EXP_NAME)

    set_up_logging_to_file(os.path.join(exp_folder, 'exp.log'))

    torch.manual_seed(47)
    device = torch.device('cuda')

    encoder = torchvision.models.vgg16_bn()
    projection = nn.Linear(1000, 10)
    model = nn.Sequential(encoder, projection)
    model = model.to(device)

    writer = SummaryWriter(exp_folder)

    logging.info('Loading datasets')

    train_dataset = Cifar10(True, get_augmentations(AUGS, random_resized_crop_size=32), device=torch.device('cpu'))
    validation_dataset = Cifar10(False, get_augmentations([], random_resized_crop_size=32), device=torch.device('cpu'))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=32, pin_memory=True)
    validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE,
                                   shuffle=False, num_workers=32, pin_memory=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LR)
    scheduler = StepLR(optimizer, step_size=LR_DECAY_EPOCHS, gamma=LR_DECAY)

    tick = 0

    for epoch in range(1, EPOCHS_NUM + 1):
        logging.info(f'Epoch {epoch}')

        writer.add_scalar('LR', scheduler.get_last_lr()[0], epoch)

        model = model.train()

        for batch_X, batch_y in tqdm(train_loader):
            optimizer.zero_grad()

            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            logits = model(batch_X)

            loss = criterion(logits, batch_y)

            writer.add_scalar('train_loss', loss.item(), tick)

            loss.backward()
            optimizer.step()

            tick += 1

        if epoch % EVALUATION_MODEL_EPOCHS == 0:
            with torch.no_grad():
                model = model.eval()

                logging.info(f'Evaluating train')
                train_accuracy = calc_accuracy(model, train_loader, device)
                logging.info(f'Evaluating validation')
                validation_accuracy = calc_accuracy(model, validation_loader, device)

                writer.add_scalars('accuracy', {
                    'train': train_accuracy,
                    'validation': validation_accuracy
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
