import cv2
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F


# val batch size = 1
class CompressorWithJPEG(nn.Module):
    def __init__(self, c1, c2, quality=10):
        super(CompressorWithJPEG, self).__init__()
        self.params = [cv2.IMWRITE_JPEG_QUALITY, quality]  # ratio:0~100

        self.cnt = 0

    def forward(self, inputs):
        b, c, h, w = inputs.shape
        inputs = inputs.reshape(h * 8, w * 16, 1)
        s = (inputs.max() - inputs.min()) / 255
        z = 255 - inputs.max() / s
        inputs_q = (inputs / s + z).type(torch.uint8)
        s = s.detach().cpu().numpy()
        z = z.detach().cpu().numpy()
        inputs_q = inputs_q.detach().cpu().numpy()
        msg = cv2.imencode(".jpg", inputs_q, self.params)[1]
        msg = (np.array(msg)).tobytes()
        if self.cnt < 10:
            print("msg:", len(inputs.detach().cpu().numpy().tobytes()) / len(msg))
            self.cnt += 1
        outputs_q = cv2.imdecode(np.frombuffer(msg, np.uint8), cv2.IMREAD_GRAYSCALE)
        outputs = (outputs_q - z) * s
        outputs = torch.tensor(outputs).to(inputs.device).reshape(b, c, h, w).type_as(inputs)
        return outputs


class BottleNetPP(nn.Module):
    def __init__(self, c1, c2, hidden_channel=128):
        super(BottleNetPP, self).__init__()

        self.conv1 = nn.Conv2d(c1, hidden_channel, kernel_size=3, stride=1, padding=1)

        self.batchnorm1 = nn.BatchNorm2d(hidden_channel)
        self.batchnorm2 = nn.BatchNorm2d(c1)

        self.conv2 = nn.ConvTranspose2d(hidden_channel, c1, kernel_size=2, stride=2)

    def forward(self, x):
        x = torch.sigmoid(self.batchnorm1(self.conv1(x)))
        x = F.relu(self.batchnorm2(self.conv2(x)))
        if self.training:
            return x, 0
        return x


class FeatureQuan(nn.Module):
    def __init__(self, c1, c2):
        super(FeatureQuan, self).__init__()

    def forward(self, x):
        s = (x.max() - x.min()) / 255
        z = 128 - x.max() / s
        x_q = (x / s + z).type(torch.int8)
        x_fp = (x_q - z) * s
        if self.training:
            return x_fp, 0
        return x_fp


class FeatureQuanOneBit(nn.Module):
    def __init__(self, c1, c2):
        super(FeatureQuanOneBit, self).__init__()

    def forward(self, x):
        x[x > 0] = 1
        x[x < 0] = 0
        if self.training:
            return x, 0
        return x


class CompressorInfer(nn.Module):
    def __init__(self, c1, c2):
        super(CompressorInfer, self).__init__()

    def forward(self, inputs):
        return inputs


if __name__ == '__main__':
    # quan = FeatureQuan(1, 2)
    # x = torch.rand(100, 100)
    # quan(x)
    cp = CompressorWithJPEG(1, 1, 100)
    x = torch.rand(1, 128, 100, 100)
    cp(x)
