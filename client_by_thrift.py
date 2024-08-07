from feature_data.format_data import Data
from feature_data.format_data import Client
from thrift.protocol import TBinaryProtocol
from thrift.transport import TTransport
from thrift.transport import TSocket
from split_model import EdgeInfer
import time


class InferClient():
    def __init__(self, weight, device='0', host='localhost', port=9000) -> None:
        self.ei = EdgeInfer(weights=weight,  device=device)
        tsocket = TSocket.TSocket(host, port)
        self.transport = TTransport.TBufferedTransport(tsocket)
        protocol = TBinaryProtocol.TBinaryProtocol(self.transport)
        self.client = Client(protocol)
        self.transport.open()
        self.t = 0.0

    def infer(self, source,i):
        pred_enc = self.ei.run(source)
        # import numpy as np
        # import sys
        # print('feats size={}KB'.format(
        #     np.array([sys.getsizeof(i) for i in pred_enc.values()]).sum()/1024))
        data = Data(**pred_enc)
        
        start = time.time()
        res = self.client.do_format(data)
        self.t += 1E3*(time.time()-start)
        print('server-answer{} time:{}'.format(res, self.t/i))


if __name__ == '__main__':
    ic = InferClient(
        'runs/train/vaenorm_one_branch_rep_r4/weights/edge.pt', device='0', host='192.168.0.103',port=9001)
    for i in range(100):
        edge_output = ic.infer('data/images/zidane.jpg',i+1)
