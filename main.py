# -*- coding: utf-8 -*-

from src.train import fnTrain
from time import time

import albumentations as A
import torch
import torch.optim as optim
from albumentations.pytorch import ToTensorV2

from src.dataloader import CreateDataloader
from src.diceloss import BCEDiceLoss
from src.model import UNET
from src.utils import CheckAccuracy, InitializeWeights

LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 50
NUM_WORKERS = 2
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_FILE_PATH = r"data/train.csv"
VALDT_FILE_PATH = r"data/test.csv"


if __name__ == "__main__":

    transformTrain = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.VerticalFlip(p=0.3),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    transformValidate = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
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

    model = UNET(in_channels=3, out_channels=1, init_features=64).to(
        DEVICE
    )

    InitializeWeights(model)

    # * single class classifiction
    # fnLoss = nn.BCEWithLogitsLoss()
    fnLoss = BCEDiceLoss()
    # fnLoss = DiceLoss()

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):
        t_epoch = time()
        print(f"### epoch: {epoch+1}/{NUM_EPOCHS}")
        epochLoss = fnTrain(
            trainLoader, DEVICE, model, optimizer, fnLoss, scaler
        )

        print(f"epoch training loss {epochLoss}")

        t_elapsed = time() - t_epoch
        print(
            f"epoch training complete in {t_elapsed//60:.0f}m {t_elapsed%60:.0f}s"
        )

        CheckAccuracy(valLoader, model)
        print("\n")
