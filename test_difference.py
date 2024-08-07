from models.yolo import Model
import argparse
from utils.general import check_yaml, print_args, set_logging
from pathlib import Path
from utils.torch_utils import select_device, time_sync
import torch
import numpy as np
from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.torch_utils import select_device, time_sync


FILE = Path(__file__).resolve()

parser = argparse.ArgumentParser()
# parser.add_argument('--cfg', type=str,
#                     default='./models/inverse/yolov5s.inverse.yaml', help='model.yaml')
parser.add_argument('--cfg', type=str,
                    default='./models/nas/yolov5s.supernet.one_branch.yaml', help='model.yaml')
parser.add_argument('--device', default='1',
                    help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
parser.add_argument('--split_choice', type=int, default=None,
                        help='split point choice')
parser.add_argument('--profile', action='store_true', help='profile model speed')

class Infer():
    def __init__(
            self,
            weights,
            imgsz=640,  # inference size (pixels)
            device='0',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
            half=False,  # use FP16 half-precision inference):
            sb=True,  # single branch
    ):
        self.weights = weights
        self.device = device
        self.sb = sb
        self.device = select_device(device)
        half &= self.device.type != 'cpu'  # half precision only supported on CUDA

        self.model = attempt_load(self.weights, map_location=self.device, fuse=False)
        self.stride = int(self.model.stride.max())  # model stride
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names  # get class names
        if half:
            self.model.half()  # to FP16
        self.half = half
        # self.model = switch_deploy(self.model)
        # print(self.model)
        inp = (1,3,imgsz,imgsz)
        #print(profile(self.model, inputs=(inp,)))
        self.model.float().eval()
        self.imgsz = check_img_size((imgsz, imgsz), s=self.stride)  # check image size
        self.pre_process_t = 0
        self.inference_t = 0
        self.post_process_t = 0
        self.cnt_t = 0

    def get_mean_time(self):
        return self.pre_process_t / self.cnt_t, self.inference_t / self.cnt_t, self.post_process_t / self.cnt_t

              
    @torch.no_grad()
    def run(self, source, split_index=None, infer=None):
        dt = [0.0, 0.0, 0.0]
        dataset = LoadImages(source, img_size=self.imgsz, stride=self.stride, auto=True)

        self.model.set_subnet(split_index)
        
        # run once
        # self.model(torch.zeros(1, 3, *self.imgsz).to(self.device).type_as(next(self.model.parameters())))
        # ratio = adaptive_scheduler(limit[0], limit[1], limit[2])
        for path, img, im0s, vid_cap in dataset:
            t1 = time_sync()
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if len(img.shape) == 3:
                img = img[None]  # expand for batch dim
            t2 = time_sync()
            dt[0] += t2 - t1
            pred, feats_origin_shape = self.model(img, augment=False, visualize=False)

            print(pred)
                  
            return {}

if __name__ == '__main__':
    opt = parser.parse_args()

    device = select_device(opt.device)
    
    model = Infer('runs/train/exp_nas_vaenorm3/weights/best.pt', device='0')

    split_index = 2

    edge_output = model.run('data/images/zidane.jpg', split_index=split_index)
    
        