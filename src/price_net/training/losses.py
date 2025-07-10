# Copied from fvcore (https://github.com/facebookresearch/fvcore)
# Licensed under the Apache License, Version 2.0
import torch
import torch.nn.functional as F


def sigmoid_focal_loss_star(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = -1,
    gamma: float = 1,
    reduction: str = "none",
) -> torch.Tensor:
    """
    FL* described in RetinaNet paper Appendix: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape. The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary classification label for each element in inputs (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance positive vs negative examples. Default = -1 (no weighting).
        gamma: Gamma parameter described in FL*. Default = 1 (no weighting).
        reduction: 'none' | 'mean' | 'sum'
            'none': No reduction will be applied to the output.
            'mean': The output will be averaged.
            'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    inputs = inputs.float()
    targets = targets.float()
    shifted_inputs = gamma * (inputs * (2 * targets - 1))
    loss = -(F.logsigmoid(shifted_inputs)) / gamma

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss *= alpha_t

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss
