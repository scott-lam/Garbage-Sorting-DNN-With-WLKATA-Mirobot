import imutils
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2

#filter the returned bounding boxes based on their probabilities
#filter boxes based on the confidence there's an object within the bbox
def filter_boxes(box_conf, boxes, box_class_probabilities, threshold = 0.5):
    box_conf_filter = box_conf[..., 0] > 0.25
    scores = box_conf[box_conf_filter]
    boxes = boxes[box_conf_filter]
    classes = box_class_probabilities[box_conf_filter]
    
    box_scores = scores * classes
    box_classes = tf.argmax(box_scores, -1)
    box_class_scores = tf.reduce_max(box_scores, -1)
    filter = box_class_scores > threshold
    scores = tf.boolean_mask(box_class_scores, filter)
    boxes = tf.boolean_mask(boxes, filter)
    classes = tf.boolean_mask(box_classes, filter)
    
    return scores, boxes, classes

#Intersection Over Union 
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

#YOLOv5 returns [x, y, w, h] of the bounding box. Must convert to xyxy so we can aply iou
def xywh2xyxy(boxes):
    # create new empty array the same shape as previous
    xyxy = np.empty_like(boxes) 
    #x and y are mid-points of bbox
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2]
    h = boxes[:, 3]

    #x1y1 should correspond to upper-left corner, x2y2 to bottom right
    xyxy[:, 0] = x - w / 2 #x1
    xyxy[:, 1] = y - h / 2 #y1
    xyxy[:, 2] = x + w / 2 #x2
    xyxy[:, 3] = y + h / 2 #y2
    
    return xyxy
    
def non_maxima_supression(scores, boxes, classes, iou_threshold = 0.5, max_boxes = 100):
    max_boxes_tensor = tf.Variable(max_boxes, dtype='int32')
    nms_indices = tf.image.non_max_suppression(boxes,scores,max_boxes_tensor, iou_threshold)

    scores = tf.gather(scores, nms_indices)
    boxes = tf.gather(boxes,nms_indices)
    classes = tf.gather(classes,nms_indices)
    
    return scores, boxes, classes

def eval(yolo_outputs, image_shape = (640, 640), max_boxes = 10, score_threshold = 0.6, iou_threshold = 0.5):
    boxes, scores, classes = detect(yolo_outputs)
    scores, boxes, classes = filter_boxes(scores, boxes, classes, threshold = score_threshold)
    #boxes = scale_boxes(boxes, image_shape)
    scores, boxes, classes = non_maxima_supression(scores, boxes, classes, iou_threshold, max_boxes)

    return scores, boxes, classes

def detect(prediction_data):
    #convert tensor size from [1, 25200, 9] => [25200, 9]
    data = prediction_data[0]
    boxes = data[:, :4] #2D array, each row contains [x y w h] of bboxes
    scores = data[:, 4:5] #2D array containing conf scores for each bbox [25200, 1]
    classes = data[:, 5:] #2D array containing class probabilities [25200, nc]
    
    boxes = xywh2xyxy(boxes)
    
    return boxes, scores, classes

def scale_boxes(boxes, image_shape):
    (image_width, image_height) = image_shape
    scaled_boxes = np.empty_like(boxes)
    scaled_boxes[:, 0] = boxes[:, 0] * image_width
    scaled_boxes[:, 1] = boxes[:, 1] * image_height
    scaled_boxes[:, 2] = boxes[:, 2] * image_width
    scaled_boxes[:, 3] = boxes[:, 3] * image_height
    
    return scaled_boxes

# From YOLOv5 repo, this is how they ressize the images 
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    img = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    print(img.shape)
    img = np.ascontiguousarray(img)
    return img

#From YOLOv5, scales the box coordinates to the relative place on the image
def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    coords = coords.numpy()
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def clip_coords(boxes, shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    if (tf.is_tensor(boxes)):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2