
ID card detection - v1 2025-10-03 10:50pm
==============================

This dataset was exported via roboflow.com on October 3, 2025 at 5:20 PM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand and search unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

For state of the art Computer Vision training notebooks you can use with this dataset,
visit https://github.com/roboflow/notebooks

To find over 100k other datasets and pre-trained models, visit https://universe.roboflow.com

The dataset includes 120 images.
ID-card are annotated in YOLOv8 Oriented Object Detection format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)

The following augmentation was applied to create 3 versions of each source image:
* Random shear of between -10° to +10° horizontally and -10° to +10° vertically
* Random Gaussian blur of between 0 and 3.9 pixels
* Salt and pepper noise was applied to 5.76 percent of pixels


