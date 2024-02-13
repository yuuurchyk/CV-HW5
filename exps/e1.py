import os
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


from cvhw5.datasets import ImageNet2012TrainSubset, ImageNet2012Validation, ImageNetEvaluation
from cvhw5.models import get_resnet50
from cvhw5.loss import get_contrastive_loss
from cvhw5.utils.logger import set_up_logger, set_up_logging_to_file
from cvhw5.utils.exp import get_exp_folder
from cvhw5.utils.score import encode, score, train_and_score


EXP_NAME = "exp1"
EPOCHS_NUM = 1000
BATCH_SIZE_TRAIN = 128
BATCH_SIZE_EVAL = 1024
LR = 1e-3
LR_DECAY = 0.95
LR_DECAY_EPOCHS = 5
AUGS = []


def main():
    exp_folder = get_exp_folder(EXP_NAME)
    device = torch.cuda()

    set_up_logger()
    set_up_logging_to_file(os.path.join(exp_folder, 'exp.log'))

    encoder = get_resnet50().to(device)
    projection = nn.Linear(1000, 128).to(device)
    model = nn.Sequential(encoder, projection)

    writer = SummaryWriter(exp_folder)

    logging.info('Loading datasets')
    train_dataset = ImageNet2012TrainSubset(1)
    train_dataset.set_augs(AUGS)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE_TRAIN,
                              collate_fn=train_dataset.collate_fn, shuffle=True)

    validation_dataset = ImageNet2012Validation()

    logging.info('Loading evaluation datasets')
    train_eval = ImageNetEvaluation(train_dataset)
    validation_eval = ImageNetEvaluation(validation_dataset)

    optimizer = Adam(model.parameters(), lr=LR)
    scheduler = StepLR(optimizer, step_size=LR_DECAY_EPOCHS, gamma=LR_DECAY, verbose=True)

    tick = 0

    for epoch in range(1, EPOCHS_NUM + 1):
        logging.info(f'Epoch {epoch}')

        for batch_X, _ in tqdm(train_loader):
            optimizer.zero_grad()

            batch_X = batch_X.to(device)

            z = model(batch_X)

            loss = get_contrastive_loss(z)

            writer.add_scalar('train_loss', loss.item(), tick)

            loss.backward()
            optimizer.step()

            tick += 1

        if epoch % LR_DECAY_EPOCHS == 0:
            logging.info(f'Evaluating')

            logging.info(f'Form embeddings for the train dataset')
            tX, ty = encode(train_eval, model, BATCH_SIZE_EVAL, device)

            logging.info(f'Form embeddings for the validation dataset')
            vX, vy = encode(validation_eval, model, BATCH_SIZE_EVAL, device)

            eval_model, train_score = train_and_score(tX, ty)
            validation_score = score(vX, vy, eval_model)

            writer.add_scalars('top1', {
                'train': train_score,
                'validation': validation_score
            }, epoch)

        scheduler.step()


if __name__ == '__main__':
    main()
