#-*-coding:utf-8-*-
from __future__ import division
import tornado.httpserver
import tornado.ioloop
import tornado.options
import tornado.web
import tornado.httpclient
import tornado.gen
from tornado.concurrent import Future

import urllib
#import json
import simplejson as json
import datetime
import hashlib
import time
from tornado.escape import json_decode, json_encode
import datetime
#from interface import interface
import requests

from head_detection_video_demo_2_2 import findPeopleNumHead
import cv2

from tornado.options import define, options
from concurrent.futures import ThreadPoolExecutor
from tornado.concurrent import run_on_executor



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
#import simplejson as json


os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

head_detector = Head_Detector_VGG16(ratios=[1], anchor_scales=[2,4])
trainer = Head_Detector_Trainer(head_detector).cuda()
trainer.load('./checkpoints/head_detector_final')



#data = {"taskId":"1", "roomId":"1", "liveUrl":"3.mp4", "recognitionDuration":30,"verifyCode":"xxxx"}
class IndexHandler(tornado.web.RequestHandler):
    executor = ThreadPoolExecutor(200)
    @tornado.web.asynchronous
    @tornado.gen.coroutine
    def post(self):
        try:
            #sends= json_decode(self.request.body)
            sends = self.request.body
            data = json.loads(self.request.body)
            liveUrl = data['liveUrl']
            recognitionDuration = data['recognitionDuration']
            roomId = data['roomId']
            print (liveUrl)

            verifyCode_str = "liveUrl="+str(data['liveUrl'])+"&recognitionDuration="+str(data['recognitionDuration'])+"&roomId="+str(data['roomId'])+"&signKey=lz123123&taskId="+str(data['taskId'])
            m = hashlib.md5()
            m.update(verifyCode_str.encode('UTF-8'))
            verifyCode = m.hexdigest()
            if verifyCode==data['verifyCode']:
                f = open('./find_num_config.json', 'r')
                find_num_config = json.loads(f.read());

                starttime = datetime.datetime.now()
                num_interrupt_cache=0
                time_interrupt_cache=0
                openurl_false_times=0
                while True:
                    endtime = datetime.datetime.now()
                    #if int((endtime-starttime).seconds) >= recognitionDuration+1000:

                        #datas = {"taskId": data['taskId'],  "averageNumber": 0, "result": "fail", "message":"cound not open liveUrl in this recognitionDuration time"}
                        #self.finish(datas)
                        #break
                    if openurl_false_times>20:
                        if num_interrupt_cache>0:
                            datas = {"taskId": data['taskId'], "averageNumber": num_interrupt_cache, "result": "success","message": "find people nums success,but not all time"}
                            self.finish(datas)
                            t.cuda.empty_cache()
                            break
                        else:
                            datas = {"taskId": data['taskId'],  "averageNumber": 0, "result": "fail", "message":"cound not open liveUrl over 20 times"}
                            self.finish(datas)
                            t.cuda.empty_cache()
                            break


                    vs = cv2.VideoCapture(liveUrl)
                    if (vs.isOpened()):
                        # url = 'http://127.0.0.1:8888'
                        # datas = {"taskId":data['taskId'], "liveUrl":liveUrl, "result": "success"}
                        # headers = {'Content-Type': 'application/json'}
                        # requests.post(url, data=json.dumps(datas), headers=headers)

                        # inters.setResponse=True
                        # inters.getResponse(sends)


                        # response ={}
                        #response = yield self.findnum(data['VideoUrl'], data['RequestDate'])
                        response = yield tornado.gen.with_timeout(datetime.timedelta(seconds=3000),self.findnum(liveUrl,data['recognitionDuration']-time_interrupt_cache),quiet_exceptions=tornado.gen.TimeoutError)
                        # tornado.ioloop.IOLoop.instance().add_callback(self.findnum(data['VideoUrl'],data['RequestDate']))
                        # self.finish("finish")

                        # url = 'http://192.168.1.252:8080/api/feedbackPeopleNumber'
                        # datas = {"taskId": data['taskId'],  "averageNumber": response['people_num'], "result": "success", "message":"find people nums success", "verifyCode":""}
                        # print datas
                        # headers = {'Content-Type': 'application/json'}
                        # requests.post(url, data=json.dumps(datas), headers=headers)

                        #verifyCode_str = "averageNumber="+str(response['people_num'])+"&message=find people nums success&result=success&signKey=lz123123&taskId=" + str(data['taskId'])
                        #m = hashlib.md5()
                        #m.update(verifyCode_str)
                        #verifyCode = m.hexdigest()
                        if abs(response['video_real_duration']+time_interrupt_cache-data['recognitionDuration'])<5:
                            if response['people_num']>num_interrupt_cache:
                                datas = {"taskId": data['taskId'], "averageNumber": response['people_num'], "result": "success","message": "find people nums success"}
                                self.finish(datas)
                                break
                            else:
                                datas = {"taskId": data['taskId'], "averageNumber": num_interrupt_cache, "result": "success","message": "find people nums success"}
                                self.finish(datas)
                                break
                        else:
                            if response['people_num']>num_interrupt_cache:
                                num_interrupt_cache=response['people_num']
                            time_interrupt_cache=time_interrupt_cache + response['video_real_duration']

                        # self.write(response)
                    else:
                        #url = 'http://192.168.41.105:8080/api/getRoomStudentLive'
                        openurl_false_times+=1
                        url = find_num_config['otherLiveUrl'] #从find_num_config.json中读取

                        #verifyCode_str = "roomId=" + roomId + "&" + "signKey=lz123123"
                        verifyCode_str = "cameraType=student&"+"roomId=" + roomId + "&" + "signKey=lz123123"
                        m = hashlib.md5()
                        m.update(verifyCode_str.encode('UTF-8'))
                        verifyCode = m.hexdigest()

                        try:
                            #datas = {"roomId": roomId, "verifyCode": verifyCode}
                            datas = {"roomId": roomId, "cameraType":"student", "verifyCode": verifyCode}
                            headers = {'Content-Type': 'application/json'}
                            r = requests.post(url, data=json.dumps(datas), headers=headers, timeout=20)
                            r = json.loads(r.text)

                            if r['result']=='success':
                                liveUrl = r['liveUrl']
                        except Exception as e:
                            pass

            else:
                #verifyCode_str = "averageNumber=0&message=verify the request fail&result=fail&signKey=lz123123&taskId=" + str(data['taskId'])
                #m = hashlib.md5()
                #m.update(verifyCode_str)
                #verifyCode = m.hexdigest()

                datas = {"taskId": data['taskId'], "averageNumber": 0, "result": "fail",
                         "message": "verify the request fail"}
                self.finish(datas)

        except Exception as e:
            print (str(e))
            traceback.print_exc()
            #verifyCode_str = "averageNumber=0&message="+str(e)+"&result=fail&signKey=lz123123&taskId=" + str(data['taskId'])
            #m = hashlib.md5()
            #m.update(verifyCode_str)
            #verifyCode = m.hexdigest()

            datas = {"taskId": data['taskId'], "averageNumber": 0, "result": "fail",
                     "message": str(e)}
            self.finish(datas)



    @run_on_executor
    def findnum(self, VideoUrl, VideoDurationTime):
        #findpeoplenum = findPeopleNumClass.findPeopleNumClass('', 0)  # 初始化find_people对象
        #findpeoplenum.video_address = VideoUrl
        #findpeoplenum.video_duration_time = VideoDurationTime

        #results = findpeoplenum.camera_recog()

        video_address = VideoUrl
        video_duration_time = VideoDurationTime
        results = findPeopleNumHead(video_address, video_duration_time,head_detector)
        response = {'url': VideoUrl, 'people_num': results[1], 'video_real_duration':results[0],'end':'****************'}

        print (results)
        return response




if __name__ == "__main__":
    #inters = interface()#初始化一个interface对象
    tornado.options.parse_command_line()
    app = tornado.web.Application(handlers=[(r"/", IndexHandler)])
    http_server = tornado.httpserver.HTTPServer(app)
    http_server.listen(8889, address="0.0.0.0")
    tornado.ioloop.IOLoop.instance().start()
