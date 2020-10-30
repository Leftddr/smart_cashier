# -*- coding: utf-8 -*-
"""
TensorFlow Object Detection API + OpenCV Sample
Created on Mon Oct 30 12:43:54 2017
@author: gchoi
"""
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import cv2
 
# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
 
from utils import label_map_util
from utils import visualization_utils as vis_util
import shutil

#test_path_folder 경로를 적는다.
path_test_folder = "C:/Users/lg/Desktop/receive/models/research/object_detection/"
TEST_FOLDER = "testset"

tf.compat.v1.reset_default_graph()
tf.compat.v1.get_default_graph()
 
# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
 
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
 
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
 
NUM_CLASSES = 90
 
opener = urllib.request.URLopener()
opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
 
for file in tar_file.getmembers():
    file_name = os.path.basename(file.name)
    if 'frozen_inference_graph.pb' in file_name:
        tar_file.extract(file, os.getcwd())

detection_graph = tf.compat.v1.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.compat.v1.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.compat.v1.import_graph_def(od_graph_def, name='')
        
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)
 
def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)
  
# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)
 
with detection_graph.as_default():
    with tf.device("/gpu:0"):
        with tf.compat.v1.Session(graph=detection_graph) as sess:
            cam = cv2.VideoCapture(0)
            while True:
                #key = input("캡처를 원하는 숫자를 누르세요 : ")
                key = 10
                reset_test_number = True
                for num in range(0, int(key)):
                    #폴더를 삭제한다.
                    if os.path.exists(path_test_folder + TEST_FOLDER + str(num)):
                        shutil.rmtree(path_test_folder + TEST_FOLDER + str(num))
                    ret_val, image = cam.read()
                    
                    if ret_val:
                        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                        image_np_expanded = np.expand_dims(image, axis=0)
                        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                        
                        # Each box represents a part of the image where a particular object was detected.
                        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                        
                        # Each score represent how level of confidence for each of the objects.
                        # Score is shown on the result image, together with the class label.
                        scores = detection_graph.get_tensor_by_name('detection_scores:0')
                        classes = detection_graph.get_tensor_by_name('detection_classes:0')
                        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                        
                        # Actual detection.
                        (boxes, scores, classes, num_detections) = sess.run(
                                [boxes, scores, classes, num_detections],
                                feed_dict={image_tensor: image_np_expanded})
                        
                        if num > 0:
                            reset_test_number = False
                        # Visualization of the results of a detection.
                        vis_util.visualize_boxes_and_labels_on_image_array(
                                image,
                                np.squeeze(boxes),
                                np.squeeze(classes).astype(np.int32),
                                np.squeeze(scores),
                                category_index,
                                use_normalized_coordinates=True,
                                line_thickness=8,
                                reset_test_number=reset_test_number,
                                end_num=int(key))
                        
                        cv2.imshow('my webcam', image)
                        
                        if cv2.waitKey(1) == 27: 
                            break  # esc to quit

                cv2.destroyAllWindows()
                break
