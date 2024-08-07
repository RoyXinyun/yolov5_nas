
from ptflops import get_model_complexity_info
from models.yolo import Model
import argparse
from utils.general import check_yaml, print_args, set_logging
from pathlib import Path
from utils.torch_utils import select_device, time_sync
import torch
FILE = Path(__file__).resolve()

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', type=str,
                    default='yolov5s.yaml', help='model.yaml')
parser.add_argument('--device', default='0',
                    help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
opt = parser.parse_args()
opt.cfg = check_yaml(opt.cfg)  # check YAML
set_logging()
device = select_device(opt.device)
model = Model(opt.cfg).to(device)

flops, params = get_model_complexity_info(
    model, (3, 640, 640), as_strings=True, print_per_layer_stat=True)
print("%s |%s" % (flops, params))

inputs = torch.ones(1, 3, 640, 640).cuda()
t1 = time_sync()
for i in range(100):
    model(inputs)
t2 = time_sync()
print('inference time: {:.2f}ms'.format((t2-t1) * 1E3/100))
