from turtle import width
import cv2
from cv2 import COLOR_BGR2GRAY
import numpy as np
from objectDetection import *
import tensorflow as tf

img = 'C:\\Users\\Scott\\Documents\\Uni\\garbage-classifcation\\camera\\opencv_frame_0.png'

can = np.array([[0.542015625, 0.5370370370370371, 0.16115625, 0.13425925925925927]])
arm = np.array([[0.5784739583333334, 0.125, 0.09344791666666666, 0.23148148148148148]])

frame = cv2.imread(img)

armCoords = xywh2xyxy(arm)
canCoords = xywh2xyxy(can)

armCoords = scale_boxes(armCoords, (1920, 1080))
canCoords = scale_boxes(canCoords, (1920, 1080))


(xmin, ymax, xmax, ymin) = armCoords[0, :]
print('Arm Coords: ' + str(armCoords[0, :]))
armCenterX = (xmin + xmax) / 2
armCenterY = (ymin + ymax) / 2
print('Center Arm: ' + str(armCenterX) + ', ' + str(armCenterY))


frame = np.ascontiguousarray(frame)
frame = cv2.rectangle(frame,(int(xmin), int(ymax)),(int(xmax), int(ymin)),(0,255,0),1)      

(xmin, ymax, xmax, ymin) = canCoords[0, :]
print('Can Coords: ' + str(canCoords[0, :]))
canCenterX = (xmin + xmax) / 2
canCenterY = (ymin + ymax) / 2
print('Center Can: ' + str(canCenterX) + ', ' + str(canCenterY))

frame = cv2.rectangle(frame,(int(xmin), int(ymax)),(int(xmax), int(ymin)),(0,255,0),1) 
cv2.putText(frame,'0 POS',(int(10), int(10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 1, cv2.LINE_AA)

cv2.imshow('frame', frame)
cv2.waitKey(0)
