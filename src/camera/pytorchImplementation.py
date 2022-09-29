#!/usr/bin/env python3

import sys
from tkinter import Frame
import numpy as np

import argparse
import torch
import cv2
import pyzed.sl as sl
import torch.backends.cudnn as cudnn
from objectDetectionPytorch import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model = torch.load('C://Users//Scott//Documents//Uni//garbage-classifcation//model//best.torchscript')
model = model.to(device)
labels = ['metal', 'glass', 'plastic', 'milk bottle']

#resize image so it matches input shape of model

def main():
    zed = sl.Camera()
    # Set configuration parameters
    input_type = sl.InputType()
    if len(sys.argv) >= 2 :
        input_type.set_from_svo_file(sys.argv[1])
    init = sl.InitParameters(input_t=input_type)
    init.camera_resolution = sl.RESOLUTION.HD1080
    init.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    init.coordinate_units = sl.UNIT.MILLIMETER

    # Open the camera
    err = zed.open(init)
    if err != sl.ERROR_CODE.SUCCESS :
        print(repr(err))
        zed.close()
        exit(1)

    # Set runtime parameters after opening the camera
    runtime = sl.RuntimeParameters()
    runtime.sensing_mode = sl.SENSING_MODE.STANDARD

    # Prepare new image size to retrieve half-resolution images
    image_size = zed.get_camera_information().camera_resolution
    image_size.width = image_size.width /2
    image_size.height = image_size.height /2

    # Declare your sl.Mat matrices
    image_zed = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.U8_C4)
    depth_image_zed = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.U8_C4)
        
    key = ' '
    while key != 32:
        if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS :
            # Retrieve the left image in sl.Mat
            zed.retrieve_image(image_zed, sl.VIEW.LEFT, sl.MEM.CPU, image_size)
            zed.retrieve_measure(depth_image_zed, sl.MEASURE.DEPTH, sl.MEM.CPU, image_size)
            # Use get_data() to get the numpy array
            image_ocv = image_zed.get_data()
            #depth_image_ocv = depth_image_zed.get_data()
            # Display the left image from the numpy array
            key = cv2.waitKey(10)
            
            frame = image_ocv[:, :, 0:3]
            #input = torch.from_numpy(frame).to(device)
            input = letterbox(frame)
            input = torch.from_numpy(input).to(device)

            tensor = input
            #tensor = torch.tensor(tensor)
            tensor = tensor.float()
            tensor /= 255
            #Expand dimension to match 4D tensor shape
            tensor = tensor[None]

            pred = model(tensor)
            scores, boxes, classes = eval(pred[0])
            boxes = scale_coords((640, 640), boxes, frame.shape)
            print('Boxes:' + str(boxes) + '\nScores: ' + str(scores) + '\nCategories: ' + str(classes))
            for i in range(len(scores)):
                score = scores[i].numpy()
                print(score)
                #box = boxes[i, :].numpy()  #convert Eager Tensor to numpy array
                box = boxes[i, :]
                print(box)
                (xmin, ymin, xmax, ymax) = box
                #print('Xmin' + str(type(xmin)) + '\nYmin: ' + str(ymin))
                label = labels[classes[i]]
                print(label)
                frame = cv2.rectangle(frame,(int(xmin), int(ymax)),(int(xmax), int(ymin)),(0,255,0),1)      
                cv2.putText(frame,label,(int(xmin), int(ymax)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 1, cv2.LINE_AA)
                cv2.putText(frame,str(score),(int(xmin), int(ymin)+10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 1, cv2.LINE_AA)

            cv2.imshow('frame', frame)

#cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
