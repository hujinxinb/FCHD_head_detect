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


from head_detection_video_demo_33 import findPeopleNumHead

head_detector = Head_Detector_VGG16(ratios=[1], anchor_scales=[2,4])
trainer = Head_Detector_Trainer(head_detector).cuda()
trainer.load('./checkpoints/head_detector_final')


class Test:
    def prt(self):
        findPeopleNumHead('./test_video/8.mp4',head_detector)
 


if __name__ == "__main__":
    t = Test()
    t.prt()
