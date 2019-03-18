#-*-coding:utf-8-*-


class interface(object):
    def __init__(self):
        self.ResponseParam = {}
        self.setResponse = False

    def setParam(self, send):
        self.ResponseParam = send

    def getParam(self):
        print self.ResponseParam
        return self.ResponseParam

    def getResponse(self, sends):
        if self.setResponse:
            self.setParam(sends)
            self.getParam()
            self.setResponse = False

        else:
            pass