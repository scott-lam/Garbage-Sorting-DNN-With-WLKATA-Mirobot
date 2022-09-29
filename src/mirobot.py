from inspect import classify_class_attrs
from mirobot import *
from time import sleep
from wlkata_mirobot import WlkataMirobot, WlkataMirobotTool
import numpy as np
import math

def setUpMirobot(x_displacement, y_displacement, z_displacement):
    arm=WlkataMirobot()
    arm.home()
    sleep(2)
    arm.set_tool_type(WlkataMirobotTool.SUCTION_CUP)
    ## Move arm out of way of camera frame
    arm.linear_interpolation(-11, -235, 195, speed=2000, wait_ok=True)
    transformationMatrix(x_displacement, y_displacement, z_displacement)
    return arm
       
def transformationMatrix(x_dis, y_dis, z_dis):
    ## displacement in mm
    ### Y displacement in the -ve direction relative to Mirobot axis ###
    dis_vec = np.array([[x_dis],
                        [-y_dis],
                        [z_dis]])

    ## BASE FRAME (0) TO CAMERA FRAME (c) ##
    ## assume clockwise rotation is positive
    x_rot_angle = np.deg2rad(180)
    z_rot_angle = np.deg2rad(-90)

    rot_mat_0_c_x = np.array([[1, 0, 0],
                            [0, np.cos(x_rot_angle), -np.sin(x_rot_angle)],
                            [0, np.sin(x_rot_angle), np.cos(x_rot_angle)]])

    rot_mat_0_c_z = np.array([[np.cos(z_rot_angle), -np.sin(z_rot_angle), 0],
                            [np.sin(z_rot_angle), np.cos(z_rot_angle), 0],
                            [0, 0, 1]])

    rot_mat = np.dot(rot_mat_0_c_x, rot_mat_0_c_z)

    extra_row = np.array([[0,0,0,1]])
    # create global homogeneous transformation matrix that can be used
    # in other functions within file
    global homo_trans_mat
    homo_trans_mat = np.concatenate((rot_mat , dis_vec), axis=1)
    homo_trans_mat = np.concatenate((homo_trans_mat, extra_row), axis=0)


def getRobotCoords(camera_frame_coords):
    coords = homo_trans_mat @ camera_frame_coords
    return np.round(coords, 1)

def cameraCoordsVector(coords):
    x = np.round((coords[0] + coords[2]) / 2, 2)
    y = np.round((coords[1] + coords[3]) / 2, 2)
    z = np.round(coords[4], 2)
    
    return np.array([x,
                     y,
                     z,
                     1])

def moveArmToObjectBin(arm, classification):
    if ((classification == 0) or (classification == 1)):
        arm.linear_interpolation(10, 220, 215, speed=2000, wait_ok=True)
        arm.pump_off()
    else:
        arm.linear_interpolation(-11, -235, 215, speed=2000, wait_ok=True)
        arm.pump_off()
            
def moveArmToObject(arm, coords, pose):
    arm.go_to_zero()
    arm.pump_on()
    ## Grabbing based on pose does not work, as the power of the suction cup is not strong enough
    ## to make a solid connection with object unless grapping from top-down
    # if (pose == 0):
    #     arm.linear_interpolation(coords[0], coords[1], coords[2], speed=2000, wait_ok=True)
    # else:
    #     if (coords[1] > 0): ## if object to the left of the base frame
    #         arm.linear_interpolation(coords[0], coords[1], coords[2], a=90, speed=2000, wait_ok=True)
    #     else:
    #         arm.linear_interpolation(coords[0], coords[1], coords[2], a=-90, speed=2000, wait_ok=True)
    arm.linear_interpolation(coords[0], coords[1], coords[2], speed=2000, wait_ok=True)
    arm.go_to_zero()

def moveRoboticArm(arm, coords, classes, poses):
    for i in range(len(coords)):
        cameraCoords = cameraCoordsVector(coords[i])
        robotCoords = getRobotCoords(cameraCoords)
        moveArmToObject(arm, robotCoords, poses[i])
        moveArmToObjectBin(arm, classes[i])
    
    