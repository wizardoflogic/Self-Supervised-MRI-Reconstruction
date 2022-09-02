import time
from collections import defaultdict
from pathlib import Path

import fastmri
import fastmri.data.transforms as T
import numpy as np
import requests
import torch
from fastmri.data import SliceDataset
from fastmri.models import Unet
from tqdm import tqdm


UNET_FOLDER = "https://dl.fbaipublicfiles.com/fastMRI/trained_models/unet/"
MODEL_FNAMES = {
    "unet_knee_sc": "knee_sc_leaderboard_state_dict.pt",
    "unet_knee_mc": "knee_mc_leaderboard_state_dict.pt",
    "unet_brain_mc": "brain_leaderboard_state_dict.pt",
}


class BaselineModule:
    """
    Wrapper class to run pretrained unet inference

    Resources
    ---------
    https://github.com/facebookresearch/fastMRI/blob/master/fastmri_examples/unet/run_pretrained_unet_inference.py
    """

    def __init__(
        self,
        data_path: Path,
        output_path: Path,
        device: str,
        state_dict_file: Path,
        challenge: str = "unet_knee_sc",
    ):
        """
        Parameters
        ----------
        data_path: Path to subsampled data
        output_path: Path for saving reconstructions
        device: Model to run
        state_dict_file: Path to saved state_dict (will download if not provided)
        challenge: Model to run
        """
        self.data_path = data_path
        self.output_path = output_path
        self.device = device
        self.state_dict_file = state_dict_file
        self.challenge = challenge

    def download_model(self, url, fname):
        response = requests.get(url, timeout=10, stream=True)

        chunk_size = 1 * 1024 * 1024  # 1 MB chunks
        total_size_in_bytes = int(response.headers.get("content-length", 0))
        progress_bar = tqdm(
            desc="Downloading state_dict",
            total=total_size_in_bytes,
            unit="iB",
            unit_scale=True,
        )

        with open(fname, "wb") as fh:
            for chunk in response.iter_content(chunk_size):
                progress_bar.update(len(chunk))
                fh.write(chunk)

        progress_bar.close()

    def run_unet_model(self, batch, model, device):
        image, _, mean, std, fname, slice_num, _ = batch

        output = model(image.to(device).unsqueeze(1)).squeeze(1).cpu()

        mean = mean.unsqueeze(1).unsqueeze(2)
        std = std.unsqueeze(1).unsqueeze(2)
        output = (output * std + mean).cpu()

        return output, int(slice_num[0]), fname[0]

    def run_inference(self):
        model = Unet(in_chans=1, out_chans=1, chans=256, num_pool_layers=4, drop_prob=0.0)
        # download the state_dict if we don't have it
        if self.state_dict_file is None:
            self.state_dict_file = self.output_path / MODEL_FNAMES[self.challenge]

        if not self.state_dict_file.exists():
            url_root = UNET_FOLDER
            self.download_model(url_root + MODEL_FNAMES[self.challenge], self.state_dict_file)

        map_location = torch.device(self.device) if self.device == "cpu" else None
        model.load_state_dict(torch.load(self.state_dict_file,map_location=map_location))
        model = model.eval()

        # data loader setup
        if "_mc" in self.challenge:
            data_transform = T.UnetDataTransform(which_challenge="multicoil")
        else:
            data_transform = T.UnetDataTransform(which_challenge="singlecoil")

        if "_mc" in self.challenge:
            dataset = SliceDataset(
                root=self.data_path,
                transform=data_transform,
                challenge="multicoil",
            )
        else:
            dataset = SliceDataset(
                root=self.data_path,
                transform=data_transform,
                challenge="singlecoil",
            )
        dataloader = torch.utils.data.DataLoader(dataset, num_workers=4)

        # run the model
        start_time = time.perf_counter()
        outputs = defaultdict(list)
        model = model.to(self.device)

        for batch in tqdm(dataloader, desc="Running inference"):
            with torch.no_grad():
                output, slice_num, fname = self.run_unet_model(batch, model, self.device)

            outputs[fname].append((slice_num, output))

        # save outputs
        for fname in outputs:
            outputs[fname] = np.stack([out for _, out in sorted(outputs[fname])])

        fastmri.save_reconstructions(outputs, self.output_path / "reconstructions")

        end_time = time.perf_counter()

        print(f"Elapsed time for {len(dataloader)} slices: {end_time-start_time}")