import enum
import cv2
from keras.backend import dtype
import numpy as np
from imutils.object_detection import non_max_suppression
import argparse
import time
import tensorflow as tf
import keras
from imutils.video import VideoStream, FPS
from tensorflow.keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from torch import classes
from objectDetection import *

model = keras.models.load_model('C://Users//Scott//Documents//Uni//garbage-classifcation//model//best_saved_model')
labels = ['metal', 'glass', 'plastic', 'milk bottle']
width = height = 640

frame = cv2.imread('C://Users//Scott//Documents//Uni//garbage-classifcation//camera//img.jpg')
print(frame.shape)
#resize image so it matches input shape of model
input = letterbox(frame)
input = input[0]

tensor = tf.convert_to_tensor(input, dtype=tf.float32)
tensor /= 255
#Expand dimension to match 4D tensor shape
tensor = np.expand_dims(tensor, axis=0)

pred = model.predict(tensor)
scores, boxes, classes = eval(pred)
boxes = scale_coords((640, 640), boxes, frame.shape)
print('Boxes:' + str(boxes) + '\nScores: ' + str(scores) + '\nCategories: ' + str(classes))
for i in range(len(scores)):
    score = scores[i].numpy()
    print(score)
    #box = boxes[i, :].numpy()  #convert Eager Tensor to numpy array
    box = boxes[i, :]
    print(box)
    (xmin, ymin, xmax, ymax) = box
    label = labels[classes[i]]
    print(label)
    frame = cv2.rectangle(frame,(int(xmin), int(ymax)),(int(xmax), int(ymin)),(0,255,0),1)      
    cv2.putText(frame,label,(int(xmin), int(ymax)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 1, cv2.LINE_AA)
    cv2.putText(frame,str(score),(int(xmin), int(ymin)+10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 1, cv2.LINE_AA)

cv2.imshow('frame', frame)
cv2.waitKey(0)

#cv2.destroyAllWindows()
