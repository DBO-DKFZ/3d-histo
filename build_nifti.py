import os
from pathlib import Path
from tqdm import tqdm

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import nibabel as nib


def build_nifti(img_p: Path, out_p: Path):
    files = sorted(list(img_p.iterdir()))
    pil_img = Image.open(files[0])
    img = np.array(pil_img)
    stack = np.zeros((len(files), *img.shape), dtype=np.uint8)  # Allocate stack array
    for i in tqdm(range(len(files))):
        pil_img = Image.open(files[i])
        img = np.array(pil_img)
        stack[i] = img
    
    # Instructions from https://stackoverflow.com/questions/40534333/how-to-write-a-color-3d-nifti-with-nibabel
    # stack is a 4-d numpy array, with the last dim holding RGB
    shape_3d = stack.shape[0:3]
    rgb_dtype = np.dtype([('R', 'u1'), ('G', 'u1'), ('B', 'u1')])
    stack = stack.copy().view(dtype=rgb_dtype).reshape(shape_3d)  # copy used to force fresh internal structure
    ni_img = nib.Nifti1Image(stack, np.eye(4))
    nib.save(ni_img, str(out_p / "3d_model.nii"))


if __name__ == "__main__":
    img_p = Path("/mnt/ssd/Data/3DTumorModell/fiji_output/convert_png")
    out_p = Path("/mnt/ssd/Data/3DTumorModell/fiji_output/nifti")
    out_p.mkdir(parents=True, exist_ok=True)
    build_nifti(img_p, out_p)
