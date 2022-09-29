import cv2
import numpy as np
import tensorflow as tf
import keras
import pyzed.sl as sl
from objectDetection import *
import sys
import statistics

model = keras.models.load_model('C://Users//Scott//Documents//Uni//garbage-classifcation//model//best_saved_model')
labels = ['metal', 'glass', 'plastic', 'milk bottle']

def getDepth(box, depth_array):
    (xmin, ymin, xmax, ymax) = box
    centerX = (xmin + xmax) / 2
    centerY = (ymin + ymax) / 2
    return depth_array.get_value(centerX, centerY)

def get_object_depth(depth, bounds):
    '''
    Calculates the median x, y, z position of top slice(area_div) of point cloud
    in camera frame.
    Arguments:
        depth: Point cloud data of whole frame.
        bounds: Bounding box for object in pixels.
            bounds[0]: x-center
            bounds[1]: y-center
            bounds[2]: width of bounding box.
            bounds[3]: height of bounding box.
    Return:
        x, y, z: Location of object in meters.
    '''
    area_div = 2

    x_vect = []
    y_vect = []
    z_vect = []

    for j in range(int(bounds[0] - area_div), int(bounds[0] + area_div)):
        for i in range(int(bounds[1] - area_div), int(bounds[1] + area_div)):
            z = depth[i, j, 2]
            if not np.isnan(z) and not np.isinf(z):
                x_vect.append(depth[i, j, 0])
                y_vect.append(depth[i, j, 1])
                z_vect.append(z)
    try:
        x_median = statistics.median(x_vect)
        y_median = statistics.median(y_vect)
        z_median = statistics.median(z_vect)
    except Exception:
        x_median = -1
        y_median = -1
        z_median = -1
        pass

    return x_median, y_median, z_median

def getPose(bbox, depth):
    displacement = 555
    if ((displacement - depth) > (bbox[0] - bbox[3])):
        return 1
    else:
        return 0

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
    
def main():
    zed, runtime, image_zed, depth_image_zed = setUpCamera()
    
    key = ' '
    while key != 27: #Press ESC key to exit
        if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS :
            # Retrieve the left image in sl.Mat
            zed.retrieve_image(image_zed, sl.VIEW.LEFT, sl.MEM.CPU)
            zed.retrieve_measure(depth_image_zed, sl.MEASURE.XYZRGBA, sl.MEM.CPU)
            # Use get_data() to get the numpy array
            image_ocv = image_zed.get_data()
            # Display the left image from the numpy array
            key = cv2.waitKey(10)
            
            frame = image_ocv[:, :, 0:3]
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
            
            print('Boxes:' + str(boxes) + '\nScores: ' + str(scores) + '\nCategories: ' + str(classes))
            for i in range(len(scores)):
                score = scores[i].numpy()
                print(score)
                box = boxes[i, :]
                print(box)
                (xmin, ymin, xmax, ymax) = box
                label = labels[classes[i]]
                print(label)
                depth = getDepth(box, depth_image_zed)
                print(depth[1])
                frame = np.ascontiguousarray(frame)
                frame = cv2.rectangle(frame,(int(xmin), int(ymax)),(int(xmax), int(ymin)),(0,255,0),1)      
                cv2.putText(frame,label,(int(xmin), int(ymax)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 1, cv2.LINE_AA)
                cv2.putText(frame,str(score),(int(xmin), int(ymin)+10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 1, cv2.LINE_AA)
                
            cv2.imshow('frame', frame)
        else:
            print(repr(zed.grab(runtime)))
            zed.close()
            exit(1)
        
if __name__ == "__main__":
    main()
    
cv2.destroyAllWindows()