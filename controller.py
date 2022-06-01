from genericpath import exists
import numpy as np
import argparse
import cv2
import os
import time
import argparse
import configparser
from face_detection import *

page_numbers=[]
parser = argparse.ArgumentParser()
parser.add_argument('--file', required=True)

args = parser.parse_args()

config = configparser.ConfigParser()
config.read("./config.ini")

cores=int(config['YOLO']['max_process'])
model_path=config['YOLO']['model_path']
img_path = args.file

s_t=time.time()
if(cores):
    set_cpu_cores(cores)
else:
    cores=cv2.getNumThreads()
get_yolo_prediction(img_path,model_path)
e_t=time.time()
model_time=round(e_t-s_t,2)

print("No of cores used:",cores)
print("Yolo Prediction Time:",model_time,"seconds")
