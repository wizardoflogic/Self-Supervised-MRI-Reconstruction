import torch
import torch.nn as nn

from pytorch_lightning.metrics import SSIM


def l1_l2_loss(
    input: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    Implement forward of normalized l1 - l2 loss.

    Parameters
    ----------
    input: Input predictions.
    target: Labels.

    Returns
    -------
    loss: Normalized L1 - L2 loss.
    """
    l1_loss_norm = torch.sum(torch.abs(input-target)) / torch.sum(torch.abs(input))
    l2_loss_norm = torch.norm(input-target) / torch.norm(input)
    
    return l1_loss_norm + l2_loss_norm


def mse_loss(
    input: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    Implement forward of MSE loss.

    Parameters
    ----------
    input: Input predictions.
    target: Labels.

    Returns
    -------
    loss: MSE loss between input and target vectors.
    """
    loss = nn.MSELoss()

    return loss(input, target)


def ssim_loss(
    input: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    Implement SSIM loss. 
    It is expensive to compute and not differentiable, so use it for only validation purpose!

    Parameters
    ----------
    input: Input predictions.
    target: Labels.

    Returns
    -------
    loss: SSIM loss between input and target vectors.
    """
    ssim = SSIM()

    return ssim(input, target)

