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

SAVE_FLAG = 1
THRESH = 0.01
IM_RESIZE = False

def read_img(path):
    f = Image.open(path)
    if IM_RESIZE:
        f = f.resize((640,480), Image.ANTIALIAS)

    f.convert('RGB')
    img_raw = np.asarray(f, dtype=np.uint8)
    print("dim2",img_raw.ndim) 
    img_raw_final = img_raw.copy()
    img = np.asarray(f, dtype=np.float32)
    D, H, W = img.shape
    
    img = img.transpose((2,0,1))
    a_D, a_H, a_W = img.shape
    img = preprocess(img)
    o_D, o_H, o_W = img.shape
    scale = o_H / H
    scale_=D/o_H
    print('D,H,W,a_D, a_H, a_W,o_D,o_H, o_W,scale:',D,H,W,a_D, a_H, a_W,o_D,o_H,o_W,scale)
    return img, img_raw_final, scale,scale_

def detect(img_path, model_path):
    file_id = utils.get_file_id(img_path)
    img, img_raw, scale,scale_ = read_img(img_path)
    print("dim1",img_raw.ndim)
    head_detector = Head_Detector_VGG16(ratios=[1], anchor_scales=[2,4])
    trainer = Head_Detector_Trainer(head_detector).cuda()
    trainer.load(model_path)
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
        utils.draw_bounding_box_on_image(image_raw,ymin*scale_, xmin*scale_, ymax*scale_, xmax*scale_)
        img_raw=np.array(image_raw)
    image_raw=Image.fromarray(np.uint8(img_raw))
    if SAVE_FLAG == 1:
       #image_raw.save('/home/hx/Project/FCHD-Fully-Convolutional-Head-Detector-master/'+file_id+'_1.png')
        image_raw.save(write_path+'/'+os.path.basename(img_path))
        frame_end = cv2.imread(write_path+'/'+os.path.basename(img_path))
        cv2.imshow("frame_end",frame_end)
        key_end = cv2.waitKey(1) & 0xFF
    else:
        image_raw.show()    


if __name__ == "__main__":
    frame_interval=10
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, help="test video path")
    parser.add_argument("--model_path", type=str, default='./checkpoints/sess:2/head_detector08120858_0.682282441835')
    args = parser.parse_args()

    
    #fps = video_capture.get(cv2.cv.CV_CAP_PROP_FPS)
    #size = (int(video_capture.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)),int(video_capture.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)))
    #videoWriter = cv2.VideoWriter(test_video.mp4,cv2.cv.CV_FOURCC('M', 'J', 'P', 'G'), fps, size)
    filename = os.path.basename(args.video_path)
    filename = os.path.splitext(filename)[0]
    write_path = './test_video_result' + '/'+filename
    if not os.path.isdir(write_path):
        os.mkdir(write_path)

    print (args.video_path)
    c = 0
    vs = cv2.VideoCapture(args.video_path)
    ret, frame = vs.read()
    while ret:
        timeF = frame_interval
        if (c % timeF == 0):
            img_path="./test_video/video_%s_%d.jpg"%(filename,c)
            cv2.imwrite(img_path,frame)
            detect(img_path, args.model_path)
            print (args.video_path,c)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
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



