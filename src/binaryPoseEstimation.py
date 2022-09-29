import numpy as np
import math

# Takes in camera coords as parameter
# coords = [x1, y1, x2, y1, z] in millimetres
# simple check, if object 'longer' than it is 'tall', position is horizontal 
# height of object found by subtracting the height of the camera to the floor from the 
# depth of the camera to the image. 
def getPoses(coords, cameraHeight):
    poses = []
    for i in range(len(coords)):
        if ((coords[i][2] - coords[i][2]) > (coords[i][4] - cameraHeight)):
            poses.append(0) ##object is horizontal
        else: 
            poses.append(1) ##object is verticle
    return poses
    
    