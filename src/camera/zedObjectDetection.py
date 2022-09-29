import numpy as np
import tensorflow as tf
import keras
import pyzed.sl as sl
from imutils.video import VideoStream, FPS
from tensorflow.keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from torch import classes
from objectDetection import *
import sys
import cv2

def main():
    zed = sl.Camera()

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720  # Use HD720 video mode
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    init_params.coordinate_units = sl.UNIT.METER
    init_params.sdk_verbose = True

    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        exit(1)

    obj_param = sl.ObjectDetectionParameters()
    obj_param.enable_tracking=True
    obj_param.image_sync=True
    obj_param.enable_mask_output=True

    camera_infos = zed.get_camera_information()
    if obj_param.enable_tracking :
        positional_tracking_param = sl.PositionalTrackingParameters()
        #positional_tracking_param.set_as_static = True
        positional_tracking_param.set_floor_as_origin = True
        zed.enable_positional_tracking(positional_tracking_param)


    print("Object Detection: Loading Module...")

    err = zed.enable_object_detection(obj_param)
    if err != sl.ERROR_CODE.SUCCESS :
        print (repr(err))
        zed.close()
        exit(1)

    objects = sl.Objects()
    obj_runtime_param = sl.ObjectDetectionRuntimeParameters()
    obj_runtime_param.detection_confidence_threshold = 10
    image_size = zed.get_camera_information().camera_resolution
    image_zed = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.U8_C4)
    
    while zed.grab() == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_image(image_zed, sl.VIEW.LEFT, sl.MEM.CPU, image_size)
        err = zed.retrieve_objects(objects, obj_runtime_param)
        
        image = image_zed.get_data()
        frame = image[:, :, 0:3]
        
        if objects.is_new :
            obj_array = objects.object_list
            print(str(len(obj_array))+" Object(s) detected\n")
            if len(obj_array) > 0 :
                first_object = obj_array[0]
                print("First object attributes:")
                print(" Label '"+repr(first_object.label)+"' (conf. "+str(int(first_object.confidence))+"/100)")
                if obj_param.enable_tracking :
                    print(" Tracking ID: "+str(int(first_object.id))+" tracking state: "+repr(first_object.tracking_state)+" / "+repr(first_object.action_state))
                position = first_object.position
                velocity = first_object.velocity
                dimensions = first_object.dimensions
                print(" 3D position: [{0},{1},{2}]\n Velocity: [{3},{4},{5}]\n 3D dimentions: [{6},{7},{8}]".format(position[0],position[1],position[2],velocity[0],velocity[1],velocity[2],dimensions[0],dimensions[1],dimensions[2]))
                if first_object.mask.is_init():
                    print(" 2D mask available")

                print(" Bounding Box 2D ")
                bounding_box_2d = first_object.bounding_box_2d
                bbox = []
                for it in bounding_box_2d :
                    print("    "+str(it),end='')
                    bbox.append(it)
                xmin = bbox[0][0]
                ymin = bbox[0][1]
                xmax = bbox[2][0]
                ymax = bbox[2][1]
                print(bbox)
                print("\n Bounding Box 3D ")
                bounding_box = first_object.bounding_box
                for it in bounding_box :
                    print("    "+str(it),end='')

                frame = np.ascontiguousarray(frame)
                frame = cv2.rectangle(frame,(int(xmin), int(ymax)),(int(xmax), int(ymin)),(0,255,0),1)      
                cv2.imshow('fram', frame)
                input('\nPress enter to continue: ')


    # Close the camera
    zed.close()

if __name__ == "__main__":
    main()
