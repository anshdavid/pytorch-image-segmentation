# -*- coding: utf-8 -*-
from typing import Optional
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from PIL import Image
import albumentations as A


class BreastCancerDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        transforms: Optional[A.Compose] = None,
        l_encoder=None,
    ):
        self.df = dataframe
        self.transforms = transforms
        self.encoder = l_encoder

        if self.encoder is not None:
            self.df["ENCODED"] = self.encoder.fit_transform(
                self.df["LABEL"]
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        imageFile = self.df.at[idx, "FILE"]
        maskFile = self.df.at[idx, "MASK"]

        image = np.array(Image.open(imageFile).convert("RGB"))
        mask = np.array(
            Image.open(maskFile).convert("L"), dtype=np.float32
        )

        mask[mask == 255] = 1.0

        if self.transforms:
            augmentaions = self.transforms(image=image, mask=mask)
            image = augmentaions["image"]
            mask = augmentaions["mask"]

        return image, mask

    def GetLabelEncoder(self):
        return self.encoder


def CreateDataloader(
    filepath: str,
    batchSize: int = 32,
    transform: Optional[A.Compose] = None,
    numWorker: int = 4,
    pinMemory: bool = True,
    shuffle: bool = True,
):

    dataset = BreastCancerDataset(
        pd.read_csv(filepath, sep=";"),
        transforms=transform,
    )

    return DataLoader(
        dataset,
        batch_size=batchSize,
        num_workers=numWorker,
        pin_memory=pinMemory,
        shuffle=shuffle,
    )
