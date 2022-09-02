import fastmri
import fastmri.data.transforms as T
import numpy as np
import torch

from data.splitter import SplitterFunc
from fastmri.data.subsample import MaskFunc
from typing import Dict, Optional, Tuple


def to_cropped_image(masked_kspace,target,attrs):
    # inverse Fourier transform to get zero filled solution
    image = fastmri.ifft2c(masked_kspace)

    # crop input to correct size
    if target is not None:
        crop_size = (target.shape[-2], target.shape[-1])
    else:
        crop_size = (attrs["recon_size"][0], attrs["recon_size"][1])

    # check for FLAIR 203
    if image.shape[-2] < crop_size[1]:
        crop_size = (image.shape[-2], image.shape[-2])

    image = T.complex_center_crop(image, crop_size)

    # absolute value
    image = fastmri.complex_abs(image)

    # normalize input
    image, mean, std = T.normalize_instance(image, eps=1e-11)
    image = image.clamp(-6, 6)

    # normalize target
    if target is not None:
        if isinstance(target,np.ndarray):
            target = T.to_tensor(target)
        target = T.center_crop(target, crop_size)
        target = T.normalize(target, mean, std, eps=1e-11)
        target = target.clamp(-6, 6)
    else:
        target = torch.Tensor([0])

    return image, target


class UnetSupervisedDataTransform(T.UnetDataTransform):
    """
    Data Transformer for training U-Net models.
    """

    def __init__(
        self,
        which_challenge: str,
        mask_func: Optional[MaskFunc] = None,
        use_seed: bool = True,
        test_mode: bool = False
    ):
        """
        Args:
            which_challenge: Challenge from ("singlecoil", "multicoil").
            mask_func: Optional; A function that can create a mask of
                appropriate shape.
            use_seed: If true, this class computes a pseudo random number
                generator seed from the filename. This ensures that the same
                mask is used for all the slices of a given volume every time.
            splitter_func
        """
        super().__init__(
            which_challenge=which_challenge,
            mask_func=mask_func,
            use_seed=use_seed
        )
        self.test_mode = test_mode

    def __call__(
        self,
        kspace: np.ndarray,
        mask: np.ndarray,
        target: np.ndarray,
        attrs: Dict,
        fname: str,
        slice_num: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, int, float]:
        """
        Args:
            kspace: Input k-space of shape (num_coils, rows, cols) for
                multi-coil data or (rows, cols) for single coil data.
            mask: Mask from the test dataset.
            target: Target image.
            attrs: Acquisition related information stored in the HDF5 object.
            fname: File name.
            slice_num: Serial number of the slice.

        Returns:
            tuple containing:
                image: Zero-filled input image.
                target: Target image converted to a torch.Tensor.
                mean: Mean value used for normalization.
                std: Standard deviation value used for normalization.
                fname: File name.
                slice_num: Serial number of the slice.
        """
        kspace = T.to_tensor(kspace)

        # check for max value
        max_value = attrs["max"] if "max" in attrs.keys() else 0.0

        # apply mask
        if self.mask_func:
            seed = None if not self.use_seed else tuple(map(ord, fname))
            masked_kspace, mask = T.apply_mask(kspace, self.mask_func, seed)
        else:
            masked_kspace = kspace

        # inverse Fourier transform to get zero filled solution
        image = fastmri.ifft2c(masked_kspace)

        if not self.test_mode:
            # crop input to correct size
            if target is not None:
                crop_size = (target.shape[-2], target.shape[-1])
            else:
                crop_size = (attrs["recon_size"][0], attrs["recon_size"][1])

        # check for FLAIR 203
        if self.test_mode or image.shape[-2] < crop_size[1]:
            crop_size = (image.shape[-2], image.shape[-2])

        image = T.complex_center_crop(image, crop_size)

        # absolute value
        image = fastmri.complex_abs(image)

        # apply Root-Sum-of-Squares if multicoil data
        if self.which_challenge == "multicoil":
            image = fastmri.rss(image)

        # normalize input
        image, mean, std = T.normalize_instance(image, eps=1e-11)
        image = image.clamp(-6, 6)

        # normalize target
        if not self.test_mode and target is not None:
            target = T.to_tensor(target)
            target = T.center_crop(target, crop_size)
            target = T.normalize(target, mean, std, eps=1e-11)
            target = target.clamp(-6, 6)
        else:
            target = torch.Tensor([0])

        return image, target, mean, std, fname, slice_num, max_value


class UnetSelfSupervisedDataTransform(T.UnetDataTransform):
    """
    Modified Data Transformer for training U-Net models.
    """

    def __init__(
        self,
        which_challenge: str,
        mask_func: Optional[MaskFunc] = None,
        use_seed: bool = True,
        splitter_func: Optional[SplitterFunc] = None,
    ):
        """
        Args:
            which_challenge: Challenge from ("singlecoil", "multicoil").
            mask_func: Optional; A function that can create a mask of
                appropriate shape.
            use_seed: If true, this class computes a pseudo random number
                generator seed from the filename. This ensures that the same
                mask is used for all the slices of a given volume every time.
            splitter_func
        """
        super().__init__(
            which_challenge=which_challenge,
            mask_func=mask_func,
            use_seed=use_seed
        )
        self.splitter_func = splitter_func

    def __call__(
        self,
        kspace: np.ndarray,
        mask: np.ndarray,
        target: np.ndarray,
        attrs: Dict,
        fname: str,
        slice_num: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, int, float]:
        """
        Args:
            kspace: Input k-space of shape (num_coils, rows, cols) for
                multi-coil data or (rows, cols) for single coil data.
            mask: Mask from the test dataset.
            target: Target image.
            attrs: Acquisition related information stored in the HDF5 object.
            fname: File name.
            slice_num: Serial number of the slice.

        Returns:
            tuple containing:
                image: Zero-filled input image.
                target: Target image converted to a torch.Tensor.
                mean: Mean value used for normalization.
                std: Standard deviation value used for normalization.
                fname: File name.
                slice_num: Serial number of the slice.
        """
        kspace = T.to_tensor(kspace)

        # check for max value
        max_value = attrs["max"] if "max" in attrs.keys() else 0.0

        # apply subsampling mask
        if self.mask_func:
            seed = None if not self.use_seed else tuple(map(ord, fname))
            masked_kspace, mask = T.apply_mask(kspace, self.mask_func, seed)
        else:
            masked_kspace = kspace

        if self.splitter_func:
            seed = None if not self.use_seed else tuple(map(ord, fname))
            mask_loss = self.splitter_func(masked_kspace.shape,seed)
        else:
            mask_loss = torch.Tensor([0])

        # normalize target
        if target is not None:
            target = T.to_tensor(target)
        else:
            target = torch.Tensor([0])

        return masked_kspace, target, fname, slice_num, max_value, attrs, mask_loss