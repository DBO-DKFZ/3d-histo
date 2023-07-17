# 3D reconstruction from histological slices

We recommend miniconda to create a virtual Python environment and install the required libraries:

`conda env create -f environment.yml`

[comment]: # (1. Export images from QuPath to .png uisng `export_images.groovy` script.)

1.) Preprocess images with OpenCV using `preprocess.py` script.

2.) Import image sequence to Fiji and run TrakEM2 plugin for spatial alignment. \
Tutorial: https://www.youtube.com/watch?v=kUyXQRcKWOk&t=2s

3.) Export aligned image sequence from TrakEM2 to .tif and convert to .png using `convert_tiff.py` script.

4.) Build 3D model from aligned .png images using `build_model.py`.