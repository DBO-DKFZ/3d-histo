import os
from pathlib import Path
from tqdm import tqdm

import numpy as np
import cv2

from PIL import Image
import matplotlib.pyplot as plt


def convert(tiff_p: Path, out_p: Path):
    paths = sorted(list(tiff_p.iterdir()))
    H_out = 3900
    W_out = 8700
    idx = 1
    for file in tqdm(paths):
        pil_img = Image.open(file)
        H_in, W_in = pil_img.height, pil_img.width
        img = np.array(pil_img)

        # compute center offset
        x_diff = (W_in - W_out) // 2
        y_diff = (H_in - H_out) // 2

        # copy img image into center of result image
        img_out = img[y_diff : y_diff + H_out, x_diff : x_diff + W_out]

        f_name = "image_%03d" % idx + ".png"
        out_f = out_p / f_name
        # plt.imsave(str(out_f), img_out)
        Image.fromarray(img_out).save(str(out_f))  # Use Pillow to write image instead of matplotlib
        idx += 1


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "-i",
        "--in_p",
        type=str,
        help="Path to exported .tif files from Fiji",
        default="/mnt/ssd/Data/3DTumorModell/fiji_output/trakem_tiff",
    )
    parser.add_argument(
        "-o",
        "--out_p",
        type=str,
        help="Path where to store coverted .png files",
        default="/mnt/ssd/Data/3DTumorModell/fiji_output/convert_png",
    )
    args = parser.parse_args()

    tiff_p = Path(args.in_p)
    out_p = Path(args.out_p)
    out_p.mkdir(parents=True, exist_ok=True)
    convert(tiff_p, out_p)
