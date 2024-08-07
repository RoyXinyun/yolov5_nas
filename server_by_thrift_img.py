# thrift -r -out ./ -gen py feature_data.thrift
from thrift.server import TServer
from thrift.protocol import TBinaryProtocol
from thrift.transport import TTransport
from thrift.transport import TSocket
from img_data import ttypes
from img_data import format_data


class FormatDataHandler(object):
    def __init__(self):
        pass

    def do_format(self, data):
        return ttypes.Result('0')


class InferServer():
    def __init__(self, host='localhost', port=9000) -> None:
        handler = FormatDataHandler()
        processor = format_data.Processor(handler)
        transport = TSocket.TServerSocket(host, port)
        # 传输方式，使用buffer
        tfactory = TTransport.TBufferedTransportFactory()
        # 传输的数据类型：二进制
        pfactory = TBinaryProtocol.TBinaryProtocolFactory()

        # 创建一个thrift 服务
        self.rpcServer = TServer.TSimpleServer(
            processor, transport, tfactory, pfactory)

        print('Starting the rpc server at', host, ':', port)
        self.rpcServer.serve()


if __name__ == "__main__":
    InferServer(device='2', host='166.111.71.49')
