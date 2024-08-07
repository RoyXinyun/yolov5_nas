from models.yolo import Model
import argparse
from utils.general import check_yaml, print_args, set_logging
from pathlib import Path
from utils.torch_utils import select_device, time_sync
import torch
import numpy as np

FILE = Path(__file__).resolve()

parser = argparse.ArgumentParser()
# parser.add_argument('--cfg', type=str,
#                     default='./models/inverse/yolov5s.inverse.yaml', help='model.yaml')
parser.add_argument('--cfg', type=str,
                    default='./models/nas/yolov5s.supernet.one_branch_vaenorm.yaml', help='model.yaml')
parser.add_argument('--device', default='0',
                    help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
parser.add_argument('--split_choice', type=int, default=None,
                        help='split point choice')
parser.add_argument('--profile', action='store_true', help='profile model speed')

opt = parser.parse_args()
opt.cfg = check_yaml(opt.cfg)  # check YAML
set_logging()
device = select_device(opt.device)
model = Model(opt.cfg).to(device)
print(model)

model.set_subnet(np.random.randint(0, 4))

# if opt.profile:
#     img = torch.rand(8 if torch.cuda.is_available() else 1, 3, 640, 640).to(device)
#     y = model(img, profile=True)


