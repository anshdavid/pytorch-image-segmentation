# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.cuda.amp.grad_scaler import GradScaler
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataloader import DataLoader


def fnTrain(
    loader: DataLoader,
    device: str,
    model: nn.Module,
    optimizer: Optimizer,
    fnLoss,
    scaler: GradScaler,
) -> float:

    runningLoss = 0
    for _, (data, targets) in enumerate(loader):

        data = data.to(device=device)
        targets = targets.float().unsqueeze(1).to(device=device)

        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = fnLoss(predictions, targets)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # print(f"batch {idxBatch+ 1} loss {loss.item()}")

        runningLoss += loss.item()

    return runningLoss / len(loader)
