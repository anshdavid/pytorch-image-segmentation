from src.diceloss import diceCoeff
import torch
import torchvision
from torch.utils.data import DataLoader


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


def CheckAccuracy(loader, model, device="cuda"):
    dice_score = 0
    num_correct = 0
    num_pixels = 0
    model.eval()

    with torch.no_grad():
        for data, target in loader:
            data = data.to(device)
            target = target.to(device).unsqueeze(1).to(device)
            preds = model(data)
            preds = (preds > 0.5).float()
            num_correct += (preds == target).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * target).sum()) / (
                (preds + target).sum() + 1e-8
            )

            torchvision.utils.save_image(preds, f"./reports/pred.png")
            torchvision.utils.save_image(target, f"./reports/truth.png")

    print(f"validation dice score: {dice_score/len(loader)}")
    # print(
    #     f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    # )
    #         dice_score += 1 - diceCoeff(preds, target)

    # print(f"validation diceCoeff {dice_score / len(loader)}")
    # model.train()
