import tensorflow as tf
import numpy as np
import cv2
import matplotlib as mpl
mpl.use('tkagg')
import matplotlib.pyplot as plt
#from cairocffi import *
import os
from os.path import join as pjoin
import sys
import copy
import detect_face
import nn4 as network
import random
import time

import sklearn

from sklearn.externals import joblib

#face detection parameters
minsize = 20 # minimum size of face
threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
factor = 0.709 # scale factor

#facenet embedding parameters

model_dir='./model_check_point/model.ckpt-500000'#"Directory containing the graph definition and checkpoint files.")
model_def= 'models.nn4'  # "Points to a module containing the definition of the inference graph.")
image_size=96 #"Image size (height, width) in pixels."
pool_type='MAX' #"The type of pooling to use for some of the inception layers {'MAX', 'L2'}.
use_lrn=False #"Enables Local Response Normalization after the first layers of the inception network."
seed=42,# "Random seed."
batch_size= None # "Number of images to process in a batch."




frame_interval=1 # frame intervals

def to_rgb(img):
  w, h = img.shape
  ret = np.empty((w, h, 3), dtype=np.uint8)
  ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
  return ret

#restore mtcnn model

print('Creating networks and loading parameters')
gpu_memory_fraction=0.3
with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess, './model_check_point/')


# obtaining frames from camera--->converting to gray--->converting to rgb
# --->detecting faces---->croping faces--->embedding--->classifying--->print


video_capture = cv2.VideoCapture('test_video_1.mp4')
fps = video_capture.get(cv2.cv.CV_CAP_PROP_FPS)
size = (int(video_capture.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)),
        int(video_capture.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)))
#videoWriter = cv2.VideoWriter('test_video_other.mp4',cv2.cv.CV_FOURCC('M', 'J', 'P', 'G'), fps, size)
# videoWriter = cv2.VideoWriter('test_video_other_1.flv', cv2.cv.CV_FOURCC('I','4','2','0'), fps, size)

c = 0
ret, frame = video_capture.read()
while ret:
    # Capture frame-by-frame
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # print(frame.shape)

    timeF = frame_interval

    if (c % timeF == 0):  # frame_interval==3, face detection every 3 frames


        find_results = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if gray.ndim == 2:
            img = to_rgb(gray)

        start = time.clock()

        bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)

        nrof_faces = bounding_boxes.shape[0]  # number of faces
        # print('The number of faces detected is{}'.format(nrof_faces))


        for face_position in bounding_boxes:

            face_position = face_position.astype(int)

            # print((int(face_position[0]), int( face_position[1])))
            # word_position.append((int(face_position[0]), int( face_position[1])))

            cv2.rectangle(frame, (face_position[0],
                                  face_position[1]),
                          (face_position[2], face_position[3]),
                          (0, 255, 0), 2)

        end = time.clock()
        time1=end-start
        print time1




        cv2.putText(frame, 'detected:{}'.format(find_results), (50, 100),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255, 0, 0),
                    thickness=2, lineType=2)


    # print(faces)

#    videoWriter.write(frame)

    cv2.imshow('Video', frame)
    ret, frame = video_capture.read()
    c += 1

    # Display the resulting frame




    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture

video_capture.release()
cv2.destroyAllWindows()
