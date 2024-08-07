import base64
import torch
import numpy as np
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.torch_utils import select_device, time_sync
from utils.datasets import LoadImages


def switch_deploy(model):
    for module in model.modules():
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()
    return model


# single branch
def encode_sb(feats, origin_shape, resize_shape):
    outs = feats['outs']
    shapes = feats['shapes']
    if 'inputs_mean' in feats.keys():
        inputs_mean = feats['inputs_mean'].cpu()
        inputs_std = feats['inputs_std'].cpu()
        inputs_mean_enc = base64.b64encode(np.array(inputs_mean, dtype="float32").tobytes()).decode('ascii')
        inputs_std_enc = base64.b64encode(np.array(inputs_std, dtype="float32").tobytes()).decode('ascii')
    origin_shape_enc = base64.b64encode(np.array(origin_shape, dtype=int).tobytes()).decode('ascii')
    resize_shape_enc = base64.b64encode(np.array(resize_shape, dtype=int).tobytes()).decode('ascii')
    tmp = outs
    tmp = (tmp / 2) + 0.5
    feats_shapes_enc = base64.b64encode(np.array(tmp.shape, dtype=int).tobytes()).decode('ascii')
    outs_enc = base64.b64encode(np.packbits(tmp.cpu().numpy().ravel().astype('bool')).tobytes()).decode('ascii')
    shapes_enc = base64.b64encode(np.array(shapes, dtype=int).tobytes()).decode('ascii')

    if 'inputs_mean' in feats.keys():
        return {
            'outs': outs_enc,
            'shapes': shapes_enc,
            'feats_shapes': feats_shapes_enc,
            'origin_shape': origin_shape_enc,
            'resize_shape': resize_shape_enc,
            'inputs_mean': inputs_mean_enc,
            'inputs_std': inputs_std_enc}
    else:
        return {
            'outs': outs_enc,
            'shapes': shapes_enc,
            'feats_shapes': feats_shapes_enc,
            'origin_shape': origin_shape_enc,
            'resize_shape': resize_shape_enc,
            'inputs_mean': 'None',
            'inputs_std': 'None'}


def decode_sb(edge_output):
    outs = edge_output['outs']
    shapes = edge_output['shapes']
    feats_shapes = edge_output['feats_shapes']
    origin_shape = edge_output['origin_shape']
    resize_shape = edge_output['resize_shape']
    if 'inputs_mean' in edge_output.keys() and edge_output['inputs_mean'] != 'None':
        inputs_mean = edge_output['inputs_mean']
        inputs_std = edge_output['inputs_std']
        inputs_mean = torch.from_numpy(np.frombuffer(base64.b64decode(inputs_mean),
                                                     dtype="float32")).cuda().unsqueeze(0).unsqueeze(2).unsqueeze(3)
        inputs_std = torch.from_numpy(np.frombuffer(base64.b64decode(inputs_std),
                                                    dtype="float32")).cuda().unsqueeze(0).unsqueeze(2).unsqueeze(3)

    origin_shape = np.frombuffer(base64.b64decode(origin_shape), dtype=int).tolist()
    resize_shape = np.frombuffer(base64.b64decode(resize_shape), dtype=int).tolist()
    shapes_dec = np.frombuffer(base64.b64decode(shapes), dtype=int).tolist()
    feats_shapes_dec = np.frombuffer(base64.b64decode(feats_shapes), dtype=int).tolist()
    feat_tmp = torch.from_numpy(np.unpackbits(np.frombuffer(base64.b64decode(outs),
                                                            dtype=np.uint8))).cuda().reshape(feats_shapes_dec).float()
    feat_tmp[feat_tmp == 0] = -1
    outs_dec = feat_tmp
    if 'inputs_mean' in edge_output.keys() and edge_output['inputs_mean'] != 'None':
        return {
            'outs': outs_dec,
            'shapes': shapes_dec,
            'feats_shapes': feats_shapes_dec,
            'inputs_mean': inputs_mean,
            'inputs_std': inputs_std}, origin_shape, resize_shape
    else:
        return {'outs': outs_dec, 'shapes': shapes_dec, 'feats_shapes': feats_shapes_dec}, origin_shape, resize_shape


# multi branch


def encode_mb(feats, origin_shape, resize_shape):
    pass


def decode_mb(edge_output):
    pass


class EdgeInfer():
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
        self.model = switch_deploy(self.model)
        print(self.model)
        self.model.float().eval()
        self.imgsz = check_img_size((imgsz, imgsz), s=self.stride)  # check image size
        self.pre_process_t = 0
        self.inference_t = 0
        self.post_process_t = 0
        self.cnt_t = 0

    def get_mean_time(self):
        return self.pre_process_t / self.cnt_t, self.inference_t / self.cnt_t, self.post_process_t / self.cnt_t

    @torch.no_grad()
    def run(self, source):
        dt = [0.0, 0.0, 0.0]
        dataset = LoadImages(source, img_size=self.imgsz, stride=self.stride, auto=True)

        for path, img, im0s, vid_cap in dataset:
            t1 = time_sync()
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if len(img.shape) == 3:
                img = img[None]  # expand for batch dim
            t2 = time_sync()
            dt[0] += t2 - t1
            pred = self.model(img, augment=False, visualize=False)
            t3 = time_sync()
            dt[1] += t3 - t2
            if self.sb:
                pred_enc = encode_sb(pred, origin_shape=im0s.shape[:2], resize_shape=img.shape[2:])
            else:
                pred_enc = encode_mb(pred, origin_shape=im0s.shape[:2], resize_shape=img.shape[2:])
            t4 = time_sync()
            dt[2] += t4 - t3

            if self.cnt_t != 0:
                self.pre_process_t += dt[0] * 1E3
                self.inference_t += dt[1] * 1E3
                self.post_process_t += dt[2] * 1E3
            self.cnt_t += 1

            # Print results
            t = tuple(x * 1E3 for x in dt)  # speeds per image
            print(
                f'\nSpeed of Edge: %.1fms pre-process, %.1fms inference, %.1fms post-process, per image at shape {(1, 3, *self.imgsz)}'
                % t)
            return pred_enc


class CloudInfer():
    def __init__(
            self,
            weights,
            imgsz=640,  # inference size (pixels)
            conf_thres=0.25,  # confidence threshold
            iou_thres=0.45,  # NMS IOU threshold
            max_det=1000,  # maximum detections per image
            device='0',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
            classes=None,  # filter by class: --class 0, or --class 0 2 3
            agnostic_nms=False,  # class-agnostic NMS
            half=False,  # use FP16 half-precision inference):
            sb=True,  # single branch
    ):
        self.weights = weights
        self.imgsz = imgsz
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.device = device
        self.classes = classes
        self.agnostic_nms = agnostic_nms
        self.sb = sb

        self.device = select_device(device)
        half &= self.device.type != 'cpu'  # half precision only supported on CUDA

        self.model = attempt_load(self.weights, map_location=self.device, fuse=False)
        self.stride = int(self.model.stride.max())  # model stride
        self.names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
            'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
            'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
            'teddy bear', 'hair drier', 'toothbrush']
        if half:
            self.model.half()  # to FP16
        self.model = switch_deploy(self.model)
        print(self.model)
        self.model.eval()
        self.imgsz = check_img_size((imgsz, imgsz), s=self.stride)  # check image size
        self.pre_process_t = 0.0
        self.inference_t = 0.0
        self.post_process_t = 0.0
        self.cnt_t = 0

    def get_mean_time(self):
        return self.pre_process_t / self.cnt_t, self.inference_t / self.cnt_t, self.post_process_t / self.cnt_t

    @torch.no_grad()
    def run(self, edge_output):
        dt = [0.0, 0.0, 0.0]
        t1 = time_sync()
        if self.sb:
            edge_output, origin_shape, resize_shape = decode_sb(edge_output)
        else:
            edge_output, origin_shape, resize_shape = decode_mb(edge_output)
        t2 = time_sync()
        dt[0] += t2 - t1
        pred = self.model(edge_output, augment=False, visualize=False)[0]
        t3 = time_sync()
        dt[1] += t3 - t2
        # NMS
        pred = non_max_suppression(pred,
                                   self.conf_thres,
                                   self.iou_thres,
                                   self.classes,
                                   self.agnostic_nms,
                                   max_det=self.max_det)
        dt[2] += time_sync() - t3
        if self.cnt_t != 0:
            self.pre_process_t += dt[0] * 1E3
            self.inference_t += dt[1] * 1E3
            self.post_process_t += dt[2] * 1E3
        self.cnt_t += 1

        # Process predictions
        s = ''
        for i, det in enumerate(pred):  # per image
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(resize_shape, det[:, :4], origin_shape).round()
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    # add to string
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "

            # Print time (inference-only)
            print(f'Result:{s}')

        # Print results
        t = tuple(x * 1E3 for x in dt)  # speeds per image
        print(
            f'Speed of Cloud: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *self.imgsz)}'
            % t)
        return det


if __name__ == '__main__':
    ei = EdgeInfer('runs/train/vaenorm_one_branch_rep_r4/weights/edge.pt', device='0')
    ci = CloudInfer('runs/train/vaenorm_one_branch_rep_r4/weights/cloud.pt', device='1')
    for i in range(100):
        edge_output = ei.run('data/images/zidane.jpg')
        ci.run(edge_output)
    print(ci.get_mean_time())
