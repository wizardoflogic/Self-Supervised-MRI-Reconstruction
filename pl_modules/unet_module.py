import fastmri
import torch

from data.transforms import to_cropped_image
from fastmri import pl_modules
from losses import l1_l2_loss


class UnetModule(pl_modules.UnetModule):
    """
    Unet training module modified to use normalized l1 - l2 loss
    """

    def __init__(self,**kwargs):
        super().__init__(**kwargs)

    def training_step(self, batch, batch_idx):
        image, target, _, _, _, _, _ = batch
        output = self(image)
        loss = l1_l2_loss(output, target)

        self.log("loss", loss.detach())

        return loss

    def validation_step(self, batch, batch_idx):
        image, target, mean, std, fname, slice_num, max_value = batch
        output = self(image)
        mean = mean.unsqueeze(1).unsqueeze(2)
        std = std.unsqueeze(1).unsqueeze(2)
        loss = l1_l2_loss(output, target)

        return {
            "batch_idx": batch_idx,
            "fname": fname,
            "slice_num": slice_num,
            "max_value": max_value,
            "output": output * std + mean,
            "target": target * std + mean,
            "val_loss": loss,
        }

    def test_step(self, batch, batch_idx):
        image, _, mean, std, fname, slice_num, _ = batch
        output = self.forward(image)
        mean = mean.unsqueeze(1).unsqueeze(2)
        std = std.unsqueeze(1).unsqueeze(2)

        return {
            "fname": fname,
            "slice": slice_num,
            "output": (output * std + mean).cpu().numpy(),
        }


class UnetSelfSupervisedModule(pl_modules.UnetModule):
    """
    Unet training module for self supervised learning
    """

    def __init__(self,**kwargs):
        super().__init__(**kwargs)

    def training_step(self, batch, batch_idx):
        subsampled_kspace, _, _, _, _, _, mask_loss = batch
        mask_train = -(mask_loss-1)

        subsampled_kspace_train = subsampled_kspace * mask_train + 0.0
        subsampled_kspace_loss = subsampled_kspace * mask_loss + 0.0

        image_train = fastmri.ifft2c(subsampled_kspace_train)
        image_train = fastmri.complex_abs(image_train)

        output_image = self(image_train)

        output_kspace = torch.fft.fft2(output_image)
        output_kspace = torch.stack((output_kspace.real, output_kspace.imag), axis=-1)

        output_kspace_loss = output_kspace * mask_loss + 0.0

        loss = l1_l2_loss(output_kspace_loss, subsampled_kspace_loss)

        self.log("loss", loss.detach())

        return loss

    def validation_step(self, batch, batch_idx):
        masked_kspace, target, fname, slice_num, max_value, attrs, _ = batch

        image, target = to_cropped_image(masked_kspace, target, attrs)

        output = self(image)

        loss = l1_l2_loss(output, target)

        return {
            "batch_idx": batch_idx,
            "fname": fname,
            "slice_num": slice_num,
            "max_value": max_value,
            "output": output,
            "target": target,
            "val_loss": loss,
        }

    def test_step(self, batch, batch_idx):
        masked_kspace, _, fname, slice_num, _, attrs, _ = batch = batch

        # image, _ = to_cropped_image(masked_kspace, None, attrs)

        image = fastmri.ifft2c(masked_kspace)
        image = fastmri.complex_abs(image)

        output = self.forward(image)

        return {
            "fname": fname,
            "slice": slice_num,
            "output": output.cpu().numpy(),
        }


class UnetSelfSupervisedContrastLearningModule(UnetSelfSupervisedModule):
    """
    Unet training module for self supervised learning with contrast learning
    """

    def __init__(self,**kwargs):
        super().__init__(**kwargs)

    def training_step(self, batch, batch_idx):
        subsampled_kspace, _, _, _, _, _, mask1 = batch
        mask2 = -(mask1-1)

        subsampled_kspace1 = subsampled_kspace * mask1 + 0.0
        subsampled_kspace2 = subsampled_kspace * mask2 + 0.0

        image1 = fastmri.ifft2c(subsampled_kspace1)
        image1 = fastmri.complex_abs(image1)
        image2 = fastmri.ifft2c(subsampled_kspace2)
        image2 = fastmri.complex_abs(image2)

        image = torch.vstack((image1, image2))

        output_image = self(image)

        output_image1 = output_image[0,:,:]
        output_image2 = output_image[1,:,:]

        output_kspace1 = torch.fft.fft2(output_image1)
        output_kspace1 = torch.stack((output_kspace1.real, output_kspace1.imag), axis=-1)
        output_kspace1 = output_kspace1 * mask2 + 0.0
        output_kspace2 = torch.fft.fft2(output_image2)
        output_kspace2 = torch.stack((output_kspace2.real, output_kspace2.imag), axis=-1)
        output_kspace2 = output_kspace2 * mask1 + 0.0

        loss = l1_l2_loss(output_kspace1, subsampled_kspace2) \
            + l1_l2_loss(output_kspace2, subsampled_kspace1)

        self.log("loss", loss.detach())

        return loss