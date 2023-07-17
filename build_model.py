import os
from pathlib import Path
from tqdm import tqdm

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

import nibabel as nib
import open3d as o3d


def build_nifti(stack: np.ndarray, out_p: Path):
    # Instructions from https://stackoverflow.com/questions/40534333/how-to-write-a-color-3d-nifti-with-nibabel
    # stack is a 4-d numpy array, with the last dim holding RGB
    shape_3d = stack.shape[0:3]
    rgb_dtype = np.dtype([("R", "u1"), ("G", "u1"), ("B", "u1")])
    stack = stack.copy().view(dtype=rgb_dtype).reshape(shape_3d)  # copy used to force fresh internal structure

    # TODO: Add correct spacing of 1µm in x/y direction and 3µm in z direction

    transform = np.array(
        ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)),
    )
    ni_img = nib.Nifti1Image(stack, affine=transform)
    print(ni_img.header)
    # ni_img.header["pixdim"] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]  # Does not help
    nib.save(ni_img, str(out_p / "3d_model_v2.nii"))


def build_ply(xyz: np.ndarray, colors: np.ndarray, out_p: Path):
    # Tutorial from http://www.open3d.org/docs/latest/tutorial/Basic/working_with_numpy.html#From-NumPy-to-open3d.PointCloud
    # Generate some neat n times 3 matrix using a variant of sync function
    # x = np.linspace(-3, 3, 401)
    # mesh_x, mesh_y = np.meshgrid(x, x)
    # z = np.sinc((np.power(mesh_x, 2) + np.power(mesh_y, 2)))
    # z_norm = (z - z.min()) / (z.max() - z.min())
    # xyz = np.zeros((np.size(mesh_x), 3))
    # xyz[:, 0] = np.reshape(mesh_x, -1)
    # xyz[:, 1] = np.reshape(mesh_y, -1)
    # xyz[:, 2] = np.reshape(z_norm, -1)

    pcd = o3d.geometry.PointCloud()
    print("Adding points to Pointcloud object")
    pcd.points = o3d.utility.Vector3dVector(xyz)
    print("Adding colors to Pointcloud object")
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)  # Colors are expected to be floats in  range [0, 1]
    print("Writing .ply file")
    o3d.io.write_point_cloud(str(out_p / "3d_model.ply"), pcd)


def main(img_p: Path, out_p: Path, scale_pct: int = 10, z_dist: int = 1):
    """
    Turn image sequence into single geometric file

    params:
    img_p: Path to image sequence
    out_p: Path where output file is stored
    scale_pct: Scaling of original image size in percent
    z_dist: Distance between each stacked image in pixels
    """

    files = sorted(list(img_p.iterdir()))
    pil_img = Image.open(files[0])
    img = np.array(pil_img)
    width = int(img.shape[1] * scale_pct / 100)
    height = int(img.shape[0] * scale_pct / 100)
    dim = (width, height)
    stack = np.zeros((len(files) * z_dist, height, width, 3), dtype=np.uint8)  # Allocate stack array
    z_val = 0
    all_xyz = []
    all_colors = []
    for i in tqdm(range(len(files))):
        pil_img = Image.open(files[i])
        img = np.array(pil_img)
        resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        # indices = np.where(np.all(img != [0, 0, 0], axis=-1))
        # xy = np.array(indices).T  # Use numpy coordinare system with x for rows and y for columns
        # xyz = np.hstack((xy, np.ones((len(xy), 1), dtype=np.uint8) * z_val))
        # colors = img[indices]
        # all_xyz.append(xyz)
        # all_colors.append(colors)
        stack[z_val] = resized
        z_val += z_dist

    stack = stack.transpose(2, 0, 1, 3)  # Transpose coordinate system to (width, depth, height, channels)
    stack = np.flip(stack, axis=2)  # Flip height Axis
    stack = np.flip(stack, axis=0)  # Flip width Axis
    stack = np.flip(stack, axis=1)  # Flip depth Axis
    build_nifti(stack, out_p)
    # all_xyz = np.vstack(all_xyz)
    # all_colors = np.vstack(all_colors)
    # build_ply(all_xyz, all_colors, out_p)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "--in_p",
        type=str,
        help="Path to image sequence",
        default="/mnt/ssd/Data/3DTumorModell/fiji_output/convert_png",
    )
    parser.add_argument(
        "--out_p",
        type=str,
        help="Path where to store 3D model file",
        default="/mnt/ssd/Data/3DTumorModell/fiji_output/nifti",
    )
    parser.add_argument(
        "--scale_pct",
        type=int,
        help="Scaling of original image size in percent",
        default=10,
    )
    parser.add_argument(
        "--z_dist",
        type=int,
        help="Distance between each stacked image in pixels",
        default=1,
    )
    args = parser.parse_args()

    img_p = Path(args.in_p)
    out_p = Path(args.out_p)
    out_p.mkdir(parents=True, exist_ok=True)
    main(img_p, out_p, scale_pct=args.scale_pct, z_dist=args.z_dist)
