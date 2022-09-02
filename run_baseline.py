"""
Script to run baseline pretrained UNet model

Resources
---------
https://github.com/facebookresearch/fastMRI/blob/master/fastmri_examples/unet/run_pretrained_unet_inference.py
"""
import utils

from pathlib import Path
from argparse import ArgumentParser, Namespace
from modules import BaselineModule


def cli_main(args: Namespace):
    model = BaselineModule(
        data_path=args.data_path,
        output_path=args.output_path,
        device=args.device,
        state_dict_file=args.state_dict_file,
    )
    model.run_inference()
    return


def build_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--config",
        default="./configs/config_baseline.yml",
        type=Path,
        help="Path to config file"
    )
    args = parser.parse_args()
    args = utils.add_args_from_config(args)
    return args


def run_cli():
    args = build_args()
    cli_main(args)


if __name__ == "__main__":
    run_cli()