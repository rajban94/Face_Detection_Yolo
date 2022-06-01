import ast
from utils import *
import numpy as np
import copy
import cv2
import configparser

config = configparser.ConfigParser()
config.read("./config.ini")

face_box_overlap_score=float(config['NMS']['overlap_score'])

def postprocess(data,img_processed_path: str,image_path):

    raw_prediction_dict=copy.deepcopy(data)
    image_bbox_dtls=copy.deepcopy(data)
    # image_cv = cv2.imread(image_path)

    face_list = ast.literal_eval(data["bbox_data"]["face_bbox"])

    #image_res = ast.literal_eval(data["page_res"])
    
    face_list = non_max_suppression(np.asarray(face_list), overlapThresh=face_box_overlap_score).tolist()

    image_bbox_dtls["bbox_data"]['face_bbox'] = str(face_list)
    
    draw_bounding_boxes(image_path,image_bbox_dtls,img_processed_path)
    return image_bbox_dtls