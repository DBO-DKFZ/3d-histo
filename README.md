# 3D reconstruction from histological slices

1.) Export images from QuPath to .png uisng `export_images.groovy` script.

2.) Preprocess images with OpenCV using `preprocess.py` script.

3.) Import image sequence to Fiji and run TrakEM2 plugin for spatial alignment. \
Tutorial: https://www.youtube.com/watch?v=kUyXQRcKWOk&t=2s

4.) Export aligned image sequence to .png from TrakEM2

5.) Build NIfTI 3D image from aligned .png images