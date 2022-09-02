import h5py
import fastmri
import matplotlib.pyplot as plt
import numpy as np

from argparse import ArgumentParser, Namespace
from fastmri.data import transforms as T
from pathlib import Path


def read_kspace_file(filepath):
    hf = h5py.File(filepath)
    kspace = hf['kspace'][()]
    kspace = T.to_tensor(kspace)
    return kspace


def visualize_kspace(kspace,dim=None,crop=False,output_filepath=None):
    kspace = fastmri.ifft2c(kspace)
    if crop:
        crop_size = (kspace.shape[-2], kspace.shape[-2])
        kspace = T.complex_center_crop(kspace, crop_size)
        kspace = fastmri.complex_abs(kspace)
        kspace, _, _ = T.normalize_instance(kspace, eps=1e-11)
        kspace = kspace.clamp(-6, 6)
    else:
        # Compute absolute value to get a real image
        kspace = fastmri.complex_abs(kspace)
    if dim is not None:
        kspace = fastmri.rss(kspace, dim=dim)
    img = np.abs(kspace.numpy())
    if output_filepath is not None:
        if not output_filepath.parent.exists():
            output_filepath.parent.mkdir(parents=True)
        plt.imshow(img, cmap='gray')
        plt.axis("off")
        plt.savefig(output_filepath,bbox_inches="tight",pad_inches=0)
    else:
        plt.imshow(img, cmap='gray')
        plt.show()


def visualize_reconstruction(filepath,output_filepath=None):
    hf = h5py.File(filepath)
    recons = hf['reconstruction'][()].squeeze()
    recons_rss = fastmri.rss(T.to_tensor(recons), dim=0)
    img = np.abs(recons_rss.numpy())
    if output_filepath is not None:
        if not output_filepath.parent.exists():
            output_filepath.parent.mkdir(parents=True)
        plt.imshow(img, cmap='gray')
        plt.axis("off")
        plt.savefig(output_filepath,bbox_inches="tight",pad_inches=0)
    else:
        plt.imshow(img, cmap='gray')
        plt.show()


def cli_main(args):
    if args.type == "reconstruction":
        if args.filepath is not None:
            visualize_reconstruction(args.filepath)
        elif args.folderpath is not None:
            for filepath in args.folderpath.glob("*.h5"):
                output_filepath = Path(f"{filepath.parent}/img/{filepath.name.replace('.h5','.jpg')}")
                if not output_filepath.exists():
                    try:
                        visualize_reconstruction(filepath,output_filepath=output_filepath)
                    except:
                        print(f"failed to visualize file: {filepath}")
    elif args.type == "kspace":
        if args.filepath is not None:
            kspace = read_kspace_file(args.filepath)
            visualize_kspace(kspace)
        elif args.folderpath is not None:
            for filepath in args.folderpath.glob("*.h5"):
                output_filepath = Path(f"{filepath.parent}/img/{filepath.name.replace('.h5','.jpg')}")
                if not output_filepath.exists():
                    kspace = read_kspace_file(filepath)
                    try:
                        visualize_kspace(kspace,dim=0,crop=True,output_filepath=output_filepath)
                    except:
                        print(f"failed to visualize file: {filepath}")


def build_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--folderpath",
        default="../output/self_supervised/reconstructions",
        type=Path,
        help="Path to folder of reconstruction .h5 files"
    )
    parser.add_argument(
        "--filepath",
        # "../data/singlecoil_train/file1000001.h5"
        default=None,
        type=Path,
        help="Path to .h5 file"
    )
    parser.add_argument(
        "--type",
        choices=("kspace","reconstruction"),
        default="reconstruction",
        type=str,
        help="Visualization type: kspace or reconstruction"
    )
    args = parser.parse_args()
    return args


def run_cli():
    args = build_args()
    cli_main(args)


if __name__ == "__main__":
    run_cli()