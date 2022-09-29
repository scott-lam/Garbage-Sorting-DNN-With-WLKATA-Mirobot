from matplotlib.pyplot import box
import tensorflow as tf
import keras
import cv2
from camera import *
from mirobot import *
from binaryPoseEstimation import getPoses
import numpy

model = tf.keras.models.load_model('./model', compile=False)
labels = ['Metal', 'Glass', 'HDPE Plastic', 'PET Plastic', 'Cardboard']

def main():
    #displacement from WLKATA Mirobot axes relative to ZED 2 axes
    x_dis = 0
    y_dis = 585
    z_dis = 564
    
    ## Set up WLKATA Mirobot and its rotation matrix
    arm = setUpMirobot(x_dis, y_dis, z_dis)
    zed, runtime, image_zed, depth_image_zed = setUpCamera()
    while True: 
        frame = videoStream(zed, runtime, image_zed, depth_image_zed)
        scores, boxes, classes = getPredictions(model, frame)
        cv2.imshow('frame', frame)
        
        if (len(classes) > 0): ##if detections
            frame = drawPredictions(frame, scores, boxes, classes, labels)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF==ord('q'): ##press q to quit
                break
            classes = classes.numpy()
            cameraCoords = getCameraCoords(boxes, depth_image_zed)
            poses = getPoses(cameraCoords, z_dis)
            moveRoboticArm(arm, cameraCoords, classes, poses)
        
        if cv2.waitKey(1) & 0xFF==ord('q'): ##press q to quit
            break
    
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()