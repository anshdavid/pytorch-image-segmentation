import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    __name__ = "dice_loss"

    def __init__(self, activation="sigmoid"):
        super(DiceLoss, self).__init__()
        self.activation = activation

    def forward(self, y_pr, y_gt):
        return 1 - diceCoeffv2(y_pr, y_gt, activation=self.activation)
        return diceCoeffv2(y_pr, y_gt, activation=self.activation)


def diceCoeff(pred, gt, smooth=1e-5, activation="sigmoid"):

    if activation is None or activation == "none":
        activation_fn = lambda x: x
    elif activation == "sigmoid":
        activation_fn = nn.Sigmoid()
    elif activation == "softmax2d":
        activation_fn = nn.Softmax2d()

    else:
        raise NotImplementedError(
            "Activation implemented for sigmoid and softmax2d activation function operation"
        )

    if activation_fn:
        pred = activation_fn(pred)

    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)

    intersection = (pred_flat * gt_flat).sum(1)
    unionset = pred_flat.sum(1) + gt_flat.sum(1)
    loss = (2 * intersection + smooth) / (unionset + smooth)

    return loss.sum() / N


def diceCoeffv2(pred, gt, eps=1e-5, activation="sigmoid"):
    """
    dice = (2 * tp) / (2 * tp + fp + fn)
    """

    if activation is None or activation == "none":
        activation_fn = lambda x: x
    elif activation == "sigmoid":
        activation_fn = nn.Sigmoid()
    elif activation == "softmax2d":
        activation_fn = nn.Softmax2d()

    else:
        raise NotImplementedError(
            "Activation implemented for sigmoid and softmax2d activation function operation"
        )

    if activation_fn:
        pred = activation_fn(pred)

    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)

    tp = torch.sum(gt_flat * pred_flat, dim=1)
    fp = torch.sum(pred_flat, dim=1) - tp
    fn = torch.sum(gt_flat, dim=1) - tp
    loss = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    return loss.sum() / N