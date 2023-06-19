import os
from pathlib import Path
from tqdm import tqdm

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import nibabel as nib
import open3d as o3d


def build_nifti(stack: np.ndarray, out_p: Path):
    # Instructions from https://stackoverflow.com/questions/40534333/how-to-write-a-color-3d-nifti-with-nibabel
    # stack is a 4-d numpy array, with the last dim holding RGB
    shape_3d = stack.shape[0:3]
    rgb_dtype = np.dtype([("R", "u1"), ("G", "u1"), ("B", "u1")])
    stack = stack.copy().view(dtype=rgb_dtype).reshape(shape_3d)  # copy used to force fresh internal structure

    # TODO: Add correct spacing of 1µm in x/y direction and 3µm in z direction

    ni_img = nib.Nifti1Image(stack, np.eye(4))
    print(ni_img.header)
    # ni_img.header["pixdim"] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]  # Does not help
    nib.save(ni_img, str(out_p / "3d_model.nii"))


def build_ply(stack: np.ndarray, out_p: Path):
    # Generate some neat n times 3 matrix using a variant of sync function
    x = np.linspace(-3, 3, 401)
    mesh_x, mesh_y = np.meshgrid(x, x)
    z = np.sinc((np.power(mesh_x, 2) + np.power(mesh_y, 2)))
    z_norm = (z - z.min()) / (z.max() - z.min())
    xyz = np.zeros((np.size(mesh_x), 3))
    xyz[:, 0] = np.reshape(mesh_x, -1)
    xyz[:, 1] = np.reshape(mesh_y, -1)
    xyz[:, 2] = np.reshape(z_norm, -1)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    o3d.io.write_point_cloud(str(out_p / "sync.ply"), pcd)


def main(img_p: Path, out_p: Path):
    files = sorted(list(img_p.iterdir()))
    pil_img = Image.open(files[0])
    img = np.array(pil_img)
    stack = np.zeros((len(files), *img.shape), dtype=np.uint8)  # Allocate stack array
    for i in tqdm(range(len(files))):
        pil_img = Image.open(files[i])
        img = np.array(pil_img)
        stack[i] = img

    # build_nifti(stack, out_p)
    build_ply(stack, out_p)


if __name__ == "__main__":
    img_p = Path("/mnt/ssd/Data/3DTumorModell/fiji_output/convert_png")
    out_p = Path("/mnt/ssd/Data/3DTumorModell/fiji_output/nifti")
    out_p.mkdir(parents=True, exist_ok=True)
    main(img_p, out_p)
