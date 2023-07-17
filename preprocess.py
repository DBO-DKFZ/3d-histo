import os
from pathlib import Path
from tqdm import tqdm

import math
import numpy as np
import cv2

from PIL import Image
import matplotlib.pyplot as plt


def my_ceil(val: int, n_decimals: int):
    return int(math.ceil(val / 10**n_decimals) * (10**n_decimals))


def tissue_detection(img: np.ndarray, remove_top_border: bool = False):
    kernel_size = 3

    # remove alpha channel
    img = img[:, :, 0:3]

    if remove_top_border:
        top_border = int(len(img) / 5)
        # hack for removing border artifacts
        img[0:top_border, :, :] = [0, 0, 0]

    # remove black background pixel
    black_px = np.where((img[:, :, 0] <= 5) & (img[:, :, 1] <= 5) & (img[:, :, 2] <= 5))
    img[black_px] = [255, 255, 255]

    # apply median filter to remove artifacts created by transitions to background pixels
    median_filtered_img = cv2.medianBlur(img, 11)

    # convert to HSV color space
    hsv_image = cv2.cvtColor(median_filtered_img, cv2.COLOR_RGB2HSV)

    # get saturation channel
    saturation = hsv_image[:, :, 1]

    # Otsu's thresholding
    _, threshold_image = cv2.threshold(saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # apply dilation to image to close spots inside mask regions
    kernel = np.ones(shape=(5, 5))
    tissue_mask = cv2.dilate(threshold_image, kernel, iterations=1)
    # tissue_mask = cv2.erode(tissue_mask, kernel)

    # find all of the connected components (white blobs in your image).
    # from https://stackoverflow.com/questions/42798659/how-to-remove-small-connected-objects-using-opencv

    # im_with_separated_blobs is an image where each detected blob has a different pixel value ranging from 1 to nb_blobs - 1.
    nb_blobs, im_with_separated_blobs, stats, _ = cv2.connectedComponentsWithStats(tissue_mask)
    # stats (and the silenced output centroids) gives some information about the blobs. See the docs for more information.
    # here, we're interested only in the size of the blobs, contained in the last column of stats.
    sizes = stats[:, -1]
    # the following lines result in taking out the background which is also considered a component, which I find for most applications to not be the expected output.
    # you may also keep the results as they are by commenting out the following lines. You'll have to update the ranges in the for loop below.
    sizes = sizes[1:]
    nb_blobs -= 1

    # minimum size of particles we want to keep (number of pixels).
    # here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever.
    top5_sizes = sorted(sizes, reverse=True)[:5]
    min_size = np.sum(top5_sizes) // len(top5_sizes)

    # output image with only the kept components
    im_result = np.zeros_like(im_with_separated_blobs, dtype=np.uint8)
    # for every component in the image, keep it only if it's above min_size
    for blob in range(nb_blobs):
        if sizes[blob] >= min_size:
            # see description of im_with_separated_blobs above
            im_result[im_with_separated_blobs == blob + 1] = 1

    tissue_mask = im_result

    return tissue_mask


def extract_tissue(img_p: Path, H_max: int, W_max: int):
    img_out = np.zeros((H_max, W_max, 3), dtype=np.uint8)
    img_in = cv2.imread(str(img_p))
    img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2RGB)
    tissue_mask = tissue_detection(img_in)
    img_in = cv2.bitwise_and(img_in, img_in, mask=tissue_mask)
    H_in, W_in, _ = img_in.shape

    # compute center offset
    x_diff = (W_max - W_in) // 2
    y_diff = (H_max - H_in) // 2

    # copy img image into center of result image
    img_out[y_diff : y_diff + H_in, x_diff : x_diff + W_in] = img_in

    # plt.imshow(img_out)
    # plt.show()

    return img_out


def main(fol_p: Path, out_p: Path):
    paths = sorted(list(fol_p.iterdir()))
    # extract_tissue(items[0])
    H_max = 0
    W_max = 0
    for img_p in paths:
        img = Image.open(img_p)  # Use Pillow due to faster performance
        H, W = img.height, img.width
        # img = cv2.imread(str(img_p))
        # H, W, C = img.shape
        if H > H_max:
            H_max = H
        if W > W_max:
            W_max = W

    H_max = my_ceil(H_max, 2)
    W_max = my_ceil(W_max, 2)

    idx = 1
    for img_p in tqdm(paths):
        f_name = "image_%03d" % idx + ".png"
        img_out = extract_tissue(img_p, H_max, W_max)
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
        help="Path to exported .png files from Qupath",
        default="/mnt/ssd/Data/3DTumorModell/qupath_output/cropped_slices",
    )
    parser.add_argument(
        "-o",
        "--out_p",
        type=str,
        help="Path where to store processed images for further spatial alignment",
        default="/mnt/ssd/Data/3DTumorModell/fiji_input/half2_png",
    )
    args = parser.parse_args()

    fol_p = Path(args.in_p)
    out_p = Path(args.out_p)
    out_p.mkdir(parents=True, exist_ok=True)
    main(fol_p, out_p)
