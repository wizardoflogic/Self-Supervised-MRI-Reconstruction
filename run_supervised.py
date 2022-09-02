"""
Script to run supervised learning

Resources
---------
https://github.com/facebookresearch/fastMRI/blob/master/fastmri_examples/unet/train_unet_demo.py
"""
import os
import pytorch_lightning as pl
import utils

from argparse import ArgumentParser, Namespace
from data.transforms import UnetSupervisedDataTransform
from fastmri.data.subsample import create_mask_for_mask_type
from fastmri.data.transforms import UnetDataTransform
from fastmri.pl_modules import FastMriDataModule
from pathlib import Path
from pl_modules import UnetModule


def cli_main(args: Namespace):
    pl.seed_everything(args.seed)

    # ------------
    # data
    # ------------
    # this creates a k-space mask for transforming input data
    mask_func = create_mask_for_mask_type(
        args.mask_type, args.center_fractions, args.accelerations
    )
    # use random masks for train transform, fixed masks for val transform
    train_transform = UnetDataTransform(args.challenge, mask_func=mask_func, use_seed=False)
    val_transform = UnetDataTransform(args.challenge, mask_func=mask_func)
    test_transform = UnetSupervisedDataTransform(args.challenge, mask_func=mask_func, test_mode=True)
    # ptl data module - this handles data loaders
    data_module = FastMriDataModule(
        data_path=args.data_path,
        challenge=args.challenge,
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=test_transform,
        test_split=args.test_split,
        test_path=args.test_path,
        sample_rate=args.sample_rate,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        distributed_sampler=(args.accelerator in ("ddp", "ddp_cpu")),
    )

    # ------------
    # model
    # ------------
    model = UnetModule(
        in_chans=args.in_chans,
        out_chans=args.out_chans,
        chans=args.chans,
        num_pool_layers=args.num_pool_layers,
        drop_prob=args.drop_prob,
        lr=args.lr,
        lr_step_size=args.lr_step_size,
        lr_gamma=args.lr_gamma,
        weight_decay=args.weight_decay,
    )

    # ------------
    # trainer
    # ------------
    trainer = pl.Trainer.from_argparse_args(args)

    # ------------
    # run
    # ------------
    if args.mode == "train":
        trainer.fit(model, datamodule=data_module)
    elif args.mode == "test":
        trainer.test(model, datamodule=data_module)
    else:
        raise ValueError(f"unrecognized mode {args.mode}")


def build_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--config",
        default="./configs/config_supervised_train.yml",
        type=Path,
        help="Path to config file"
    )
    # trainer config
    parser = pl.Trainer.add_argparse_args(parser)
    parser.set_defaults(
        gpus=2,  # number of gpus to use
        replace_sampler_ddp=False,  # this is necessary for volume dispatch during val
        accelerator="ddp",  # what distributed version to use
        seed=42,  # random seed
        deterministic=True,  # makes things slower, but deterministic
        default_root_dir=None,  # directory for logs and checkpoints
        max_epochs=50,  # max number of epochs
    )

    args = parser.parse_args()

    # add args from .yml config
    args = utils.add_args_from_config(args)

    if not hasattr(args, "test_path"):
        args.test_path = None

    # configure checkpointing in checkpoint_dir
    checkpoint_dir = args.default_root_dir / "checkpoints"
    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir(parents=True)

    args.checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=checkpoint_dir,
        save_top_k=True,
        verbose=True,
        monitor="validation_loss",
        mode="min",
    )

    # set default checkpoint if one exists in our checkpoint directory
    if not hasattr(args, "resume_from_checkpoint"):
        ckpt_list = sorted(checkpoint_dir.glob("*.ckpt"), key=os.path.getmtime)
        if ckpt_list:
            args.resume_from_checkpoint = str(ckpt_list[-1])
        else:
            args.resume_from_checkpoint = None

    print(f"resuming from checkpoint: {args.resume_from_checkpoint}")

    return args


def run_cli():
    args = build_args()
    cli_main(args)


if __name__ == "__main__":
    run_cli()