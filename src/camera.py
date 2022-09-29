from cmath import isinf, isnan
import cv2
import numpy as np
import tensorflow as tf
import pyzed.sl as sl
from objectDetection import *
import sys
import math

def getDepth(box, depth_array):
    (xmin, ymin, xmax, ymax) = box
    centerX = (xmin + xmax) / 2
    centerY = (ymin + ymax) / 2
    depth = depth_array.get_value(centerX, centerY)[1]
    #Sometimes depth can return a NaN value, due to shadows, lighting conditions etc
    #check if depth is nan, and if so, iterate through the area of the bbox
    #until a depth is found
    if (isnan(depth) or isinf(depth)):
        depth = []
        for i in range(int(xmin), int(xmax), 50):
            for j in range(int(ymin), int(ymax), 50):
                    depth.append(depth_array.get_value(i, j)[1])
        depth = min(depth)
    return np.round(depth, 2)

# Return camera object, image and depth matricies, runtime parameters
def setUpCamera():
    zed = sl.Camera()
    
    # Set configuration parameters
    input_type = sl.InputType()
    if len(sys.argv) >= 2 :
        input_type.set_from_svo_file(sys.argv[1])
    init = sl.InitParameters(input_t=input_type)
    init.camera_resolution = sl.RESOLUTION.HD1080
    init.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    init.coordinate_units = sl.UNIT.MILLIMETER
    init.depth_minimum_distance = 20
    
    err = zed.open(init)
    if err != sl.ERROR_CODE.SUCCESS :
        print(repr(err))
        zed.close()
        exit(1)

    # Set runtime parameters after opening the camera
    runtime = sl.RuntimeParameters()
    runtime.sensing_mode = sl.SENSING_MODE.STANDARD

    # Prepare image size to define matricies
    image_size = zed.get_camera_information().camera_resolution

    # Declare your sl.Mat matrices
    image_zed = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.U8_C4)
    depth_image_zed = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.U8_C4)
    
    return zed, runtime, image_zed, depth_image_zed
       
def getPredictions(model, frame):
    input = letterbox(frame)
    input = input[0]

    # convert array to tensor and normalise pixel values
    tensor = tf.convert_to_tensor(input, dtype=tf.float32)
    tensor /= 255
    
    #Expand dimension to match 4D tensor shape
    tensor = np.expand_dims(tensor, axis=0)

    pred = model.predict(tensor)
    scores, boxes, classes = eval(pred)
    boxes = scale_coords((640, 640), boxes, frame.shape)
    
    return scores, boxes, classes

def drawPredictions(frame, scores, boxes, classes, labels):
    for i in range(len(scores)):
        score = scores[i].numpy()
        #print(score)
        box = boxes[i, :]
        #print(box)
        (xmin, ymin, xmax, ymax) = box
        label = labels[classes[i]]
        #print(label)
        frame = np.ascontiguousarray(frame)
        frame = cv2.rectangle(frame,(int(xmin), int(ymax)),(int(xmax), int(ymin)),(0,255,0),1)      
        cv2.putText(frame,label,(int(xmin), int(ymax)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 1, cv2.LINE_AA)
        cv2.putText(frame,str(score),(int(xmin), int(ymin)+10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 1, cv2.LINE_AA)
    
    return frame

## Creates an array containing the bouding box of the image, and its depth at the centre
## return bbox and depth coords in millimetres
def getCameraCoords(boxes, depth_image_zed):
    cameraCoords = []
    for i in range(len(boxes)):
        box = boxes[i, :]
        depth = getDepth(box, depth_image_zed)
        box = bbox_to_mm(box)
        cameraCoords.append(np.append(box, depth))
    return cameraCoords

def videoStream(zed, runtime, image_zed, depth_image_zed):
    if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS :
        # Retrieve the left image in sl.Mat
        zed.retrieve_image(image_zed, sl.VIEW.LEFT, sl.MEM.CPU)
        zed.retrieve_measure(depth_image_zed, sl.MEASURE.DEPTH, sl.MEM.CPU)
        # Use get_data() to get the numpy array
        image_ocv = image_zed.get_data()
        #convert 4 channel image into 3 channel RGB image for inference
        frame = image_ocv[:, :, 0:3]
    else:
        print(repr(zed.grab(runtime)))
        zed.close()
        exit(1)
        
    return frame
        
        
def bbox_to_mm(bbox):
    mm_per_pixel = 1035/1920
    # x_mm = (( bbox[0] + bbox[2]) / 2 ) * mm_per_pixel
    # y_mm = ((bbox[1] + bbox[3]) / 2) * mm_per_pixel
    
    # return np.round(x_mm, 2), np.round(y_mm, 2)
    
    x1_mm = bbox[0] * mm_per_pixel
    x2_mm = bbox[2] * mm_per_pixel
    y1_mm = bbox[1] * mm_per_pixel
    y2_mm = bbox[3] * mm_per_pixel
    
    return x1_mm, y1_mm, x2_mm, y2_mm
