#-*-coding:utf-8-*-

from flask import Flask, jsonify
from flask import render_template, redirect,url_for
from flask import request
import requests
#import json
import simplejson as json
import datetime
import hashlib

url = 'http://127.0.0.1:8889'


#verifyCode_str = "liveUrl="+str("./test_video/3.mp4")+"&recognitionDuration="+str(30)+"&roomId="+str(1)+"&signKey=lz123123&taskId="+str(2)
#m = hashlib.md5()
#m.update(verifyCode_str.encode('UTF-8'))
verifyCode = "16ef775006cbc2a134267993e51d2f15"
data = {"taskId":"2", "roomId":"1", "liveUrl":"./test_video/3.mp4","recognitionDuration":30,"verifyCode":verifyCode}

headers = {'Content-Type' : 'application/json'}

r = requests.post(url, data=json.dumps(data), headers=headers)
print (r.text)
r = json.loads(r.text)

print (r['result'])



# if r['url']=='1.mp4':
#     print r.text
# else:
#     print "error"


