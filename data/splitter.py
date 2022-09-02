import numpy as np
import torch

from typing import Optional, Sequence, Tuple, Union


class SplitterFunc:
    """
    SplitterFunc creates a mask to split the indices into two sets

    mask = mask_train represents the kspace locations to be used for training.
    ~mask = mask_loss represents the kspace locations to be used in the loss function.
    """

    def __init__(
        self,
        loss_ratio: float,
        splitter_type: str,
    ):
        """
        Args:
            loss_ratio: Ratio of kspace locations to be used for the loss function.
        """
        self.loss_ratio = loss_ratio
        self.rng = np.random.RandomState()  # pylint: disable=no-member
        self.mode = splitter_type  # Uses uniform distribution by default

    def __call__(
        self,
        shape: Tuple[int],
        seed: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> torch.Tensor:

        if len(shape) < 3:
            raise ValueError("Shape should have 3 or more dimensions")

        self.rng.seed(seed)

        if self.mode == "gaussian":
            mask = self.rng.normal(0.5, 0.5, size=shape) < self.loss_ratio
        else:
            mask = self.rng.uniform(0, 1, size=shape) < self.loss_ratio

        mask = torch.from_numpy(mask.astype(np.float32))

        return mask

