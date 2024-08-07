from split_model import EdgeInfer, CloudInfer
from utils.torch_utils import time_sync

infer = EdgeInfer(
    'runs/train/vaenorm_one_branch_rep_r4/weights/edge.pt', device='0')
cloud_infer = CloudInfer(
    'runs/train/vaenorm_one_branch_rep_r4/weights/cloud.pt', device='0')
for _ in range(5):
    t1 = time_sync()
    edge_output = infer.run('data/images/zidane.jpg')
    # response = requests.post(
    #     'http://127.0.0.1:5000/predict', json=edge_output)
    # import sys
    # print('feats size={}KB'.format(
    #     np.array([sys.getsizeof(i) for i in edge_output['outs']]).mean()/1024))
    # print('shapes size={}KB'.format(
    #     np.array([sys.getsizeof(i) for i in edge_output['shapes']]).mean()/1024))
    # print('feats_shapes size={}KB'.format(
    #     np.array([sys.getsizeof(i) for i in edge_output['feats_shapes']]).mean()/1024))
    det = cloud_infer.run(edge_output)
    # print('edge time:', (time_sync() - t1)*1E3)
