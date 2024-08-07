from models.yolo import Model
import torch
from utils.torch_utils import intersect_dicts, de_parallel
from copy import deepcopy


def convert_model(cfg, origin_ckp, save_path, infer_type):

    model = Model(cfg, ch=3, nc=80, anchors=None, infer_type=infer_type).half()  # create
    model.cuda()
    ckpt = torch.load(origin_ckp)
    # final epoch
    csd = ckpt['model'].state_dict()
    # csd = ckpt['ema'].state_dict()
    print('model stride:', model.stride)
    print('ckpt stride:', ckpt['model'].stride)
    csd = intersect_dicts(csd, model.state_dict(), exclude=['anchor'])  # intersect
    # breakpoint()
    model.load_state_dict(csd, strict=False)  # load

    print(model)
    ckpt = {'model': deepcopy(de_parallel(model)).half()}

    # Save last, best and delete
    torch.save(ckpt, save_path)


# cfg_edge = 'models/vae/edge_cloud/yolov5s.vae.one_branch.te.r4.edge.yaml'
# cfg_cloud = 'models/vae/edge_cloud/yolov5s.vae.one_branch.te.r4.cloud.yaml'
# origin_ckp = 'runs/train/vae_one_branch_te_r4/weights/last.pt'
# save_path_edge = 'runs/train/vae_one_branch_te_r4/weights/edge.pt'
# save_path_cloud = 'runs/train/vae_one_branch_te_r4/weights/cloud.pt'

cfg_edge = 'models/vae/edge_cloud/yolov5s.compressor.one_branch.edge.yaml'
cfg_cloud = 'models/vae/edge_cloud/yolov5s.compressor.one_branch.cloud.yaml'

origin_ckp = 'runs/train/quan/weights/last.pt'
save_path_edge = 'runs/train/quan/weights/edge.pt'
save_path_cloud = 'runs/train/quan/weights/cloud.pt'

convert_model(cfg_edge, origin_ckp, save_path_edge, 'edge')
convert_model(cfg_cloud, origin_ckp, save_path_cloud, 'cloud')
