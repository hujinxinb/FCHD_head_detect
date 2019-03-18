from __future__ import division

import os
import torch as t
from src.config import opt
from src.head_detector_vgg16 import Head_Detector_VGG16
from trainer import Head_Detector_Trainer
from PIL import Image
import numpy as np
from data.dataset import preprocess
import matplotlib.pyplot as plt 
import src.array_tool as at
from src.vis_tool import visdom_bbox
import argparse
import src.utils as utils
from src.config import opt
import time
import cv2

from align_custom import AlignCustom
from face_feature import FaceFeature
from mtcnn_detect import MTCNNDetect
from tf_graph import FaceRecGraph
import argparse
import sys
import json



from head_detection_video_demo_55 import findPeopleNumHead

FRGraph = FaceRecGraph();
face_detect = MTCNNDetect(FRGraph, scale_factor=2);  # scale_factor, rescales image for faster detection


head_detector = Head_Detector_VGG16(ratios=[1], anchor_scales=[2,4])
trainer = Head_Detector_Trainer(head_detector).cuda()
trainer.load('./checkpoints/head_detector_final')





class Test:
    def prt(self):
        findPeopleNumHead('./test_video/1.mp4',head_detector,face_detect)
 


if __name__ == "__main__":
    t = Test()
    t.prt()
