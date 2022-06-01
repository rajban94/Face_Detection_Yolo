import numpy as np
import cv2
import os
import configparser
from utils import *
from processor import *

def set_cpu_cores(cores):
    set_cpu_cores_util(cores)
    cv2.setNumThreads(cores)
config = configparser.ConfigParser()
config.read("./config.ini")



def get_raw_prediction(net,img_path,target_image_res):
    confidence_thresh=float(config['YOLO']['confidence'])
    img_width=int(config['YOLO']['img_width'])
    img_height=int(config['YOLO']['img_width'])
    img_res=(img_width,img_height)
    labels=['face']
    layer_names = net.getLayerNames()
    layer_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    image = cv2.imread(img_path)
    image_bbox_dtls= make_prediction(net, layer_names, image,confidence_thresh,img_res,target_image_res)
    final_json={}
    final_json["bbox_data"]=image_bbox_dtls
    final_json["page_res"]=str(target_image_res)
    return final_json

def load_model(model_path):
    weights_path=model_path
    cfg_path=model_path.replace(".weights",".cfg")
    net = cv2.dnn.readNetFromDarknet(cfg_path,weights_path)
    return net

def get_yolo_prediction(img_path,model_path):
    net=load_model(model_path)
    im_name=os.path.basename(img_path)

    if(not os.path.exists("final_output")):
            os.mkdir("final_output")
    folder_path= "final_output/"+im_name

    img_width=int(config['YOLO']['img_width'])
    img_height=int(config['YOLO']['img_width'])
    image_res=(img_width,img_height)
    img_processed_path=folder_path+"/"+im_name+"_processed_prediction.jpg"
    image_bbox_dtls = get_raw_prediction(net,img_path,image_res)
    processed_dtls = postprocess(image_bbox_dtls,img_processed_path,img_path)
    return processed_dtls