#-*-coding:utf-8-*-
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

from head_detection_video_demo_2 import findPeopleNumHead
import cv2

from tornado.options import define, options
from concurrent.futures import ThreadPoolExecutor
from tornado.concurrent import run_on_executor



#data = {"taskId":"1", "roomId":"1", "liveUrl":"3.mp4", "recognitionDuration":30,"verifyCode":"xxxx"}
class IndexHandler(tornado.web.RequestHandler):
    executor = ThreadPoolExecutor(50)
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

                while True:
                    endtime = datetime.datetime.now()
                    if int((endtime-starttime).seconds) >= recognitionDuration:

                        #verifyCode_str = "averageNumber=0&message=cound not open liveUrl in this recognitionDuration time&result=fail&signKey=lz123123&taskId="+str(data['taskId'])
                        #m = hashlib.md5()
                        #m.update(verifyCode_str)
                        #verifyCode = m.hexdigest()

                        datas = {"taskId": data['taskId'],  "averageNumber": 0, "result": "fail", "message":"cound not open liveUrl in this recognitionDuration time"}
                        self.finish(datas)
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
                        response = yield tornado.gen.with_timeout(datetime.timedelta(seconds=500),self.findnum(data['liveUrl'],data['recognitionDuration']),quiet_exceptions=tornado.gen.TimeoutError)
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
                        datas = {"taskId": data['taskId'], "averageNumber": response['people_num'], "result": "success","message": "find people nums success"}
                        self.finish(datas)
                        break

                        # self.write(response)
                    else:
                        #url = 'http://192.168.41.105:8080/api/getRoomStudentLive'
                        url = find_num_config['otherLiveUrl'] #从find_num_config.json中读取

                        verifyCode_str = "roomId=" + roomId + "&" + "signKey=lz123123"
                        m = hashlib.md5()
                        m.update(verifyCode_str.encode('UTF-8'))
                        verifyCode = m.hexdigest()

                        try:
                            datas = {"roomId": roomId, "verifyCode": verifyCode}
                            headers = {'Content-Type': 'application/json'}
                            r = requests.post(url, data=json.dumps(datas), headers=headers, timeout=8)
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
        results = findPeopleNumHead(video_address, video_duration_time)
        response = {'url': VideoUrl, 'people_num': results[1], 'end':'****************'}

        print (results)
        return response




if __name__ == "__main__":
    #inters = interface()#初始化一个interface对象
    tornado.options.parse_command_line()
    app = tornado.web.Application(handlers=[(r"/", IndexHandler)])
    http_server = tornado.httpserver.HTTPServer(app)
    http_server.listen(8889, address="0.0.0.0")
    tornado.ioloop.IOLoop.instance().start()
