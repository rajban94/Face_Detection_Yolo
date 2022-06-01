# Face_Detection_Yolo

Prerequisites
Tensorflow
opencv-python

There are many ways to install virtual environment (virtualenv), see the Python Virtual Environments: A Primer guide for different platforms, but here are a couple:

For Ubuntu

     pip install virtualenv
For Mac

     pip install --upgrade virtualenv

Create a Python 3.6 virtual environment for this project and activate the virtualenv:

    virtualenv -p python3.6 yoloface

    source ./yoloface/bin/activate

Usage
Clone this repository

     git clone https://github.com/rajban94/Face_Detection_Yolo/
For face detection, you should download the YOLOv3 weights file

Run the following command:

image input

    python controller.py --file 'image path'
