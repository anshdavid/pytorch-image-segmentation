# -*- coding: utf-8 -*-

from src.diceloss import DiceLoss
from time import time

import albumentations as A
import torch
import torch.nn as nn
import torch.optim as optim
from albumentations.pytorch import ToTensorV2
from torch.cuda.amp.grad_scaler import GradScaler
from torch.optim.optimizer import Optimizer

from src.dataloader import CreateDataloader
from src.model import UNET
from src.utils import CheckAccuracy

LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 50
NUM_WORKERS = 2
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_FILE_PATH = r"data/train.csv"
VALDT_FILE_PATH = r"data/test.csv"


def fnTrain(
    loader,
    model: nn.Module,
    optimizer: Optimizer,
    fnLoss,
    scaler: GradScaler,
) -> float:

    runningLoss = 0
    for idxBatch, (data, targets) in enumerate(loader):

        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = fnLoss(predictions, targets)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # print(f"batch {idxBatch+ 1} loss {loss.item()}")

        runningLoss += loss.item()

    return runningLoss


def run():

    transformTrain = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    transformValidate = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    trainLoader = CreateDataloader(
        TRAIN_FILE_PATH,
        batchSize=BATCH_SIZE,
        transform=transformTrain,
        numWorker=NUM_WORKERS,
        pinMemory=PIN_MEMORY,
        shuffle=True,
    )
    valLoader = CreateDataloader(
        VALDT_FILE_PATH,
        batchSize=BATCH_SIZE,
        transform=transformValidate,
        numWorker=NUM_WORKERS,
        pinMemory=PIN_MEMORY,
        shuffle=False,
    )

    model = UNET(in_channels=3, out_channels=1).to(DEVICE)

    # * single class classifiction
    fnLoss = nn.BCEWithLogitsLoss()
    # fnLoss = DiceLoss()

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):
        t_epoch = time()
        print(f"### epoch: {epoch+1}/{NUM_EPOCHS}")
        epochLoss = fnTrain(trainLoader, model, optimizer, fnLoss, scaler)

        print(f"epoch training loss {epochLoss / len(trainLoader)}")

        t_elapsed = time() - t_epoch
        print(
            f"epoch training complete in {t_elapsed//60:.0f}m {t_elapsed%60:.0f}s"
        )

        CheckAccuracy(valLoader, model)
        print("\n")


if __name__ == "__main__":
    run()
