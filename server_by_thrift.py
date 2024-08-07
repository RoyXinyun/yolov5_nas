# thrift -r -out ./ -gen py feature_data.thrift
from thrift.server import TServer
from thrift.protocol import TBinaryProtocol
from thrift.transport import TTransport
from thrift.transport import TSocket
from feature_data import ttypes
from feature_data import format_data
from split_model import CloudInfer


class FormatDataHandler(object):
    def __init__(
        self,
        weight,
        device='0',
    ):
        self.ci = CloudInfer(weight, device=device)

    def do_format(self, data):
        data_ = {
            'outs': data.outs,
            'shapes': data.shapes,
            'feats_shapes': data.feats_shapes,
            'origin_shape': data.origin_shape,
            'resize_shape': data.resize_shape,
            'inputs_mean': data.inputs_mean,
            'inputs_std': data.inputs_std
        }
        det = self.ci.run(data_)
        print(det)
        return ttypes.Result('0')


class InferServer():
    def __init__(self,
                 weight,
                 device='0',
                 host='localhost',
                 port=9000) -> None:
        handler = FormatDataHandler(weight, device)
        processor = format_data.Processor(handler)
        transport = TSocket.TServerSocket(host, port)
        # 传输方式，使用buffer
        tfactory = TTransport.TBufferedTransportFactory()
        # 传输的数据类型：二进制
        pfactory = TBinaryProtocol.TBinaryProtocolFactory()

        # 创建一个thrift 服务
        self.rpcServer = TServer.TSimpleServer(processor, transport, tfactory,
                                               pfactory)

        print('Starting the rpc server at', host, ':', port)
        self.rpcServer.serve()


if __name__ == "__main__":
    InferServer('runs/train/vaenorm_one_branch_rep_r4/weights/cloud.pt',
                device='2',
                host='166.111.71.49')
