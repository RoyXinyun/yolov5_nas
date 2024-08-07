from img_data.format_data import Data
from img_data.format_data import Client
from thrift.protocol import TBinaryProtocol
from thrift.transport import TTransport
from thrift.transport import TSocket
import time
import cv2
import base64
import numpy as np
from PIL import Image

class InferClient():
    def __init__(self, host='localhost', port=9000) -> None:
        tsocket = TSocket.TSocket(host, port)
        self.transport = TTransport.TBufferedTransport(tsocket)
        protocol = TBinaryProtocol.TBinaryProtocol(self.transport)
        self.client = Client(protocol)
        self.transport.open()
        self.t = 0.0

    def read_img(self, source,i):
        img = Image.open(source)
        inputs_mean_enc = base64.b64encode(np.array(img, dtype="uint8").tobytes()).decode('ascii')
        # import sys
        # print('feats size={}KB'.format(sys.getsizeof(inputs_mean_enc)/1024))
        data = Data(inputs_mean_enc)
        
        start = time.time()
        res = self.client.do_format(data)
        self.t += 1E3*(time.time()-start)
        print('server-answer{} time:{}'.format(res, self.t/i))


if __name__ == '__main__':
    ic = InferClient(host='192.168.0.103')
    for i in range(100):
        edge_output = ic.read_img('data/images/zidane.jpg',i+1)
