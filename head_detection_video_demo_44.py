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
import PIL.ImageDraw as ImageDraw
import PIL.Image as Image



SAVE_FLAG = 1
THRESH = 0.01
IM_RESIZE = False

show=1


def read_img(img_read):
    if IM_RESIZE:
        img_read = cv2.resize(img_read,(640,480), interpolation=cv2.INTER_CUBIC)
    img_raw = np.asarray(img_read, dtype=np.uint8)
    print("dim2",img_raw.ndim) 
    img_raw_final = img_raw.copy()
    img = np.asarray(img_read, dtype=np.float32)
    D, H, W = img.shape
    
    img = img.transpose((2,0,1))
    a_D, a_H, a_W = img.shape
    img = preprocess(img)
    o_D, o_H, o_W = img.shape
    scale = o_H / H
    scale_=D/o_H
    print('D,H,W,a_D, a_H, a_W,o_D,o_H, o_W,scale:',D,H,W,a_D, a_H, a_W,o_D,o_H,o_W,scale)
    return img, img_raw_final, scale,scale_

def detect(img_read, write_path,head_detector,img_path):
    global show
    img, img_raw, scale,scale_ = read_img(img_read)
    print("dim1",img_raw.ndim)
    
    img_raw_2=img_raw.copy()
    img = at.totensor(img)
    img = img[None, : ,: ,:]
    img = img.cuda().float()
    st = time.time()
    pred_bboxes_, _ = head_detector.predict(img, scale, mode='evaluate', thresh=THRESH)
    et = time.time()
    tt = et - st
    print ("[INFO] Head detection over. Time taken: {:.4f} s".format(tt))
    for i in range(pred_bboxes_.shape[0]):
        print(i)
        ymin, xmin, ymax, xmax = pred_bboxes_[i,:]
        print(ymin, xmin, ymax, xmax)
        image_raw=Image.fromarray(np.uint8(img_raw))
        image_raw_2 = Image.fromarray(np.uint8(img_raw_2))

        utils.draw_bounding_box_on_image(image_raw,ymin*scale_, xmin*scale_, ymax*scale_, xmax*scale_)
        #cv2.rectangle(image_raw_2, (int(xmin*scale_), int(ymin*scale_)), (int(xmax*scale_), int(ymax*scale_)), 255,cv2.FILLED)
        #draw_rect = ImageDraw.Draw(image_raw_2)
        #draw_rect.rectangle((xmin*scale_, ymin*scale_, xmax*scale_, ymax*scale_), 'red')

        img_raw=np.array(image_raw)
        img_raw_2=np.array(image_raw_2)
    image_raw=Image.fromarray(np.uint8(img_raw))
    image_raw_2=Image.fromarray(np.uint8(img_raw_2))

    image_raw = cv2.cvtColor(np.asarray(image_raw),cv2.COLOR_RGB2BGR)
    image_raw_2 = cv2.cvtColor(np.asarray(image_raw_2),cv2.COLOR_RGB2BGR)



    if SAVE_FLAG == 1:
       #image_raw.save('/home/hx/Project/FCHD-Fully-Convolutional-Head-Detector-master/'+file_id+'_1.png')
       #image_raw.save(write_path+'/'+os.path.basename(img_path))
       #frame_end = cv2.imread(write_path+'/'+os.path.basename(img_path))
       #cv2.imshow("frame_end",frame_end)
        cv2.imwrite(write_path+'/'+os.path.basename(img_path),image_raw)
        cv2.imshow("image_raw",image_raw)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            show=0
    else:
        image_raw.show()    


def findPeopleNumHead(video_address, head_detector,face_detect):
    global show
    frame_interval=5

    
    #fps = video_capture.get(cv2.cv.CV_CAP_PROP_FPS)
    #size = (int(video_capture.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)),int(video_capture.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)))
    #videoWriter = cv2.VideoWriter(test_video.mp4,cv2.cv.CV_FOURCC('M', 'J', 'P', 'G'), fps, size)
    #filename = os.path.basename(args.video_path)
    #filename = os.path.splitext(filename)[0]
    filename = "test2222"
    write_path = './test_video_result' + '/'+filename
    if not os.path.isdir(write_path):
        os.mkdir(write_path)

    c = 0
    vs = cv2.VideoCapture(video_address)
    ret, frame = vs.read()
    while ret and show:
        timeF = frame_interval
        if (c % timeF == 0):
            img_path="./test_video/video_%s_%d.jpg"%(filename,c)

        rects, landmarks = face_detect.detect_face(frame, 20)
        for (i, rect) in enumerate(rects):
            cv2.rectangle(frame, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), 255,cv2.FILLED)

            img_read = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            #cv2.imwrite(img_path,frame)
            detect(img_read, write_path,head_detector,img_path)
            print (video_address,c)
        ret, frame = vs.read()
        c+=1

    # model_path = './checkpoints/sess:2/head_detector08120858_0.682282441835'

    # test_data_list_path = os.path.join(opt.data_root_path, 'brainwash_test.idl')
    # test_data_list = utils.get_phase_data_list(test_data_list_path)
    # data_list = []
    # save_idx = 0
    # with open(test_data_list_path, 'rb') as fp:
    #     for line in fp.readlines():
    #         if ":" not in line:
    #             img_path, _ = line.split(";")
    #         else:
    #             img_path, _ = line.split(":")

    #         src_path = os.path.join(opt.data_root_path, img_path.replace('"',''))
    #         detect(src_path, model_path, save_idx)
    #         save_idx += 1



