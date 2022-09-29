import cv2
import numpy as np
import pyzed.sl as sl
import sys

zed = sl.Camera()
# Set configuration parameters
input_type = sl.InputType()
if len(sys.argv) >= 2 :
    input_type.set_from_svo_file(sys.argv[1])
init = sl.InitParameters(input_t=input_type)
init.camera_resolution = sl.RESOLUTION.HD1080

# Open the camera
err = zed.open(init)
if err != sl.ERROR_CODE.SUCCESS :
    print(repr(err))
    zed.close()
    exit(1)

# Set runtime parameters after opening the camera
runtime = sl.RuntimeParameters()
# Prepare new image size to retrieve half-resolution images
image_size = zed.get_camera_information().camera_resolution
# image_size.width = image_size.width /2
# image_size.height = image_size.height /2

# Declare your sl.Mat matrices
image_zed = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.U8_C4)

if zed.grab() == sl.ERROR_CODE.SUCCESS :
    # Retrieve the left image in sl.Mat
    zed.retrieve_image(image_zed, sl.VIEW.LEFT)
    # Use get_data() to get the numpy array
    image_ocv = image_zed.get_data()
    # Display the left image from the numpy array
    
img_counter = 0

while True:
    if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS :
            # Retrieve the left image in sl.Mat
            zed.retrieve_image(image_zed, sl.VIEW.LEFT, sl.MEM.CPU, image_size)
            # Use get_data() to get the numpy array
            image_ocv = image_zed.get_data()
            cv2.imshow('frame', image_ocv)
            print(image_ocv.shape)
            print(image_size)
            k = cv2.waitKey(1)
            if k%256 == 27:
                # ESC pressed
                print("Escape hit, closing...")
                break
            elif k%256 == 32:
                # SPACE pressed
                img_name = "opencv_frame_{}.png".format(img_counter)
                cv2.imwrite(img_name, image_ocv)
                print("{} written!".format(img_name))
                img_counter += 1


zed.close()
cv2.destroyAllWindows()