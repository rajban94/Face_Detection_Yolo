import cv2
import numpy as np
import os
import ast

def get_bb_box(bb_box):
    x, y = bb_box[0], bb_box[1]
    w, h = bb_box[2], bb_box[3]
    return [x,y,x+w,y+h]

def non_max_suppression(boxes: np.array, overlapThresh: float = 0.4) -> np.array:
    """
    To ignore smaller overlapping bounding boxes
    and return only larger ones.

    Parameters:
        boxes: np.array
               List of all bounding boxes
        overlapThresh: float
                       area of the current smallest region
                       divided by the area of current bounding box
    

    Return:
        Array of bounding boxes whithout overlapping boxes.
    """

    if len(boxes) == 0:
        return []
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    pick = []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / area[idxs[:last]]
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))
    return boxes[pick].astype("int")

def draw_bounding_boxes(image_path,image_bbox_dtls,img_save_path):
    image_path=image_path
    processed_image=cv2.imread(image_path)
    bb_box_dtls=image_bbox_dtls["bbox_data"]    
    color_dict={"face_bbox":(255,0,0)}

    for key,value in bb_box_dtls.items():
        if(value and key!="cell_bbox"):
            value=ast.literal_eval(value)
            for each_value in value:
                boxes=each_value
                x1, y1 = boxes[0], boxes[1]
                x2, y2 = boxes[2], boxes[3]
                #conf = boxes[4]
                processed_image = cv2.rectangle(processed_image, (x1, y1), (x2, y2), color_dict[key], 2)

    processed_image = cv2.putText(processed_image, 'no of face- '+str(len(value)), (0,45), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (255, 0, 0), 3, cv2.LINE_AA)

    img_save_path = "C:/Users/User/Desktop/pythonProject/Face_Detection_Yolo/final_output"
    cv2.imwrite(img_save_path+'/'+'result_'+os.path.basename(image_path),processed_image)

#### face detection ####

def extract_boxes_confidences_classids(outputs, confidence, width, height):
    boxes = []
    confidences = []
    classIDs = []

    for output in outputs:
        for detection in output:
            # Extract the scores, classid, and the confidence of the prediction
            scores = detection[5:]
            classID = np.argmax(scores)
            conf = scores[classID]

            # Consider only the predictions that are above the confidence threshold
            if conf > confidence:
                # Scale the bounding box back to the size of the image
                box = detection[0:4] * np.array([width, height, width, height])
                centerX, centerY, w, h = box.astype('int')

                # Use the center coordinates, width and height to get the coordinates of the top left corner
                x = int(centerX - (w / 2))
                y = int(centerY - (h / 2))

                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(conf))
                classIDs.append(classID)
    return boxes, confidences, classIDs

def make_prediction(net, layer_names, image,confidence_thresh,image_res,target_res):

    image_bbox_dtls = {
        'face_bbox': []
        }
    
    height, width = image.shape[:2]
    # Create a blob and pass it through the model
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, image_res, swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(layer_names)
    # Extract bounding boxes, confidences and classIDs
    boxes, confidences, classIDs = extract_boxes_confidences_classids(outputs, confidence_thresh, width, height)

    face_list=[]
    face_conf=[]

    if len(classIDs) > 0:
        for index,class_id in enumerate(classIDs):
            
            face_list.append(get_bb_box(boxes[index]))
            face_conf.append(int(round(confidences[index],2)*100))
    
    image_bbox_dtls["face_bbox"]=str(face_list)
    return image_bbox_dtls