import keras
from transformers import BertForMaskedLM
from objectDetection import *
from camera import *
import os
import glob
import cv2
import numpy
from sklearn.metrics import accuracy_score

model = keras.models.load_model('./best_saved_model')
labels = ['Metal', 'Glass', 'HDPE Plastic', 'PET Plastic', 'Cardboard']
dir = 'C:\\Users\\Scott\\Documents\\Uni\\garbage-classifcation\\images\\Recycling Images\\test\\images'
labelDir = 'C:\\Users\\Scott\\Documents\\Uni\\garbage-classifcation\\images\\Recycling Images\\test\\labels'
saveDir = './detect/test/'
cwd = os.getcwd()
number = 0
        
def inferenceImages(predictionsArray, saveDir):
    for file in os.listdir(dir):
        frame = cv2.imread(dir + '\\' + file)
        input = letterbox(frame)
        input = input[0]
        tensor = tf.convert_to_tensor(input, dtype=tf.float32)
        tensor /= 255
        #Expand dimension to match 4D tensor shape
        tensor = np.expand_dims(tensor, axis=0)

        pred = model.predict(tensor)
        scores, boxes, classes = eval(pred)
        boxes = scale_coords((640, 640), boxes, frame.shape)
        for i in range(len(scores)):
            score = scores[i].numpy()
            box = boxes[i, :]
            (xmin, ymin, xmax, ymax) = box
            label = labels[classes[i]]
            predictionsArray.append(classes[i].numpy())
            frame = cv2.rectangle(frame,(int(xmin), int(ymax)),(int(xmax), int(ymin)),(0,255,0),1)      
            cv2.putText(frame,label,(int(xmin), int(ymax)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 1, cv2.LINE_AA)
            cv2.putText(frame,str(score),(int(xmin), int(ymin)+10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 1, cv2.LINE_AA)
            cv2.imwrite(os.path.join(saveDir, file), frame)
    return predictionsArray

def getTrueLabels(truePredictions):
    for file in os.listdir(labelDir):
        f = open(labelDir + '\\' + file)
        text = f.readlines()
        for line in text:
            truePredictions.append(line[0])
    return truePredictions

def main():
    number = 0
    saveDir = './detect/test/'
    while True:
        saveDir += str(number)
        if not os.path.exists(saveDir):
            os.makedirs(saveDir)
            break
        else:
            number = number + 1
            
    predictionsArray = []
    inferenceImages(predictionsArray, saveDir)
            


def iou(box1, box2):
    #Get coordinates of corner points of Intersecting bouding box
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection_area = ((y2 - y1) * (x2 - x1))
    box1_area = (box1[3]-box1[1])*(box1[2]-box1[0])
    box2_area = (box2[3]-box2[1])*(box2[2]-box2[0])
    union_area = box1_area + box2_area - intersection_area
    iou = intersection_area/union_area
    
    return iou

if __name__ == '__main__':
    main()