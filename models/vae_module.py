import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F
from .repconv import RepConv
import numpy as np
from .soca import SOCA
from .enhance_conv import invertedBlock, ConvFFN


class SignFunction(Function):
    def __init__(self):
        super(SignFunction, self).__init__()

    @staticmethod
    def forward(ctx, input):
        return input.sign()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class Sign(nn.Module):
    def __init__(self):
        super(Sign, self).__init__()

    def forward(self, x):
        return SignFunction.apply(x)


class Binarizer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Binarizer, self).__init__()
        self.sign = Sign()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.tanh(x)
        return self.sign(x)


class Identity(nn.Module):
    def __init__(self, out_channl, id):
        super(Identity, self).__init__()
        self.id = id

    def forward(self, x):
        if self.id == -1:
            return x
        return x[self.id]

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert len(size) == 4
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization_new(content_feat, style_mean, style_std):
    size = content_feat.size()
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(
        size
    )
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)

class VAEModule(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=0, g=1, r=1, encoder_type=0):
        super(VAEModule, self).__init__()
        
        self.choice_block = nn.ModuleList([])
        self.choice_block.append(Identity(c2, -1))
        if encoder_type == 0:
            self.choice_block.append(VAE(c2, k, s, p, g, r))
        elif encoder_type == 1:
            self.choice_block.append(VAENorm(c2, k, s, p, g, r))
        elif encoder_type == 2:
            self.choice_block.append(FeatureQuanOneBit(c1, c2))
        
    def forward(self, x, choice):
        # print(self.choice_block[choice])
        if choice == 1 and self.training: 
            x, loss = self.choice_block[choice](x)
            return x, loss
        else:
            x = self.choice_block[choice](x)
            return x
        
class FeatureQuanOneBit(nn.Module):
    def __init__(self, c1, c2):
        super(FeatureQuanOneBit, self).__init__()
        self.binarizer = Binarizer(c1, c2)

    def forward(self, x):
        x = self.binarizer(x)
        if self.training:
            return x, torch.tensor(0)
        return x

class Encoder(nn.Module):
    def __init__(self, dim, k=2, s=1, p=0, reduction=1, groups=1):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(dim, dim, 1, 1)
        self.act1 = nn.SiLU()
        self.bn1 = nn.BatchNorm2d(dim)
        self.conv2 = nn.Conv2d(dim, dim // reduction, k, stride=s, padding=p, groups=groups)
        self.act2 = nn.SiLU()
        self.bn2 = nn.BatchNorm2d(dim // reduction)
        self.binarizer = Binarizer(dim // reduction, dim // reduction)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.bn2(x)
        x = self.binarizer(x)
        return x

class Decoder(nn.Module):
    def __init__(self, dim, k=2, s=1, p=0, reduction=1, groups=1):
        super(Decoder, self).__init__()
        self.conv1 = nn.ConvTranspose2d(dim // reduction, dim, k, stride=s, padding=p, groups=groups)
        self.bn1 = nn.BatchNorm2d(dim)
        self.act1 = nn.SiLU()
        self.conv2 = nn.Conv2d(dim, dim, 1, 1)
        self.bn2 = nn.BatchNorm2d(dim)
        self.act2 = nn.SiLU()

        # self.inverted_block = invertedBlock(dim, dim)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)

        # x = self.inverted_block(x)

        return x


# class Encoder(nn.Module):
#     def __init__(self, dim, k=2, s=1, p=0, reduction=1, groups=1):
#         super(Encoder, self).__init__()
#         self.enc = nn.Sequential(
#             nn.Conv2d(dim, dim, 1, 1),
#             nn.SiLU(),
#             nn.BatchNorm2d(dim),
#             nn.Conv2d(dim, dim // reduction, k, stride=s, padding=p, groups=groups),
#             nn.SiLU(),
#             nn.BatchNorm2d(dim // reduction),
#         )
#         self.binarizer = Binarizer(dim // reduction, dim // reduction)

#     def forward(self, x):
#         x = self.enc(x)
#         x = self.binarizer(x)
#         return x

# class Decoder(nn.Module):
#     def __init__(self, dim, k=2, s=1, p=0, reduction=1, groups=1):
#         super(Decoder, self).__init__()
#         self.dec = nn.Sequential(
#             nn.ConvTranspose2d(
#                 dim // reduction, dim, k, stride=s, padding=p, groups=groups),
#             nn.BatchNorm2d(dim),
#             nn.SiLU(),
#             nn.Conv2d(dim, dim, 1, 1),
#             nn.BatchNorm2d(dim),
#             nn.SiLU(),
#         )

#         # self.inverted = invertedBlock(dim, dim)
#         # self.soca = SOCA(dim)
#         # self.convffn = ConvFFN(dim, dim)

#     def forward(self, x):
#         x = self.dec(x)
#         # x = self.soca(x)
#         # x = self.inverted(x)
#         return x

class VAE(nn.Module):
    def __init__(self, dim, k=2, s=1, p=0, groups=1, reduction=1):
        super(VAE, self).__init__()
        self.enc = Encoder(dim, k, s, p, reduction, groups)
        self.dec = Decoder(dim, k, s, p, reduction, groups)

        # self.soca = SOCA(dim)
        # self.convffn = ConvFFN(dim, dim)
        
        self.criteria = nn.SmoothL1Loss()
        
    def forward(self, x):
        losses = 0
        shapes = [x.shape[2:]]

        # inputs = F.interpolate(
        #     x,
        #     size=(int(shapes[0][0] * 1), int(shapes[0][1] * 1)),
        #     mode="bilinear",
        #     align_corners=True,
        # )
        # inputs_mean, inputs_std = calc_mean_std(inputs)
        # # standarlization
        # inputs = (inputs - inputs_mean.expand(inputs.size())) / inputs_std.expand(
        #     inputs.size()
        # )

        out = self.enc(x)
        out = self.dec(out)

        out = F.interpolate(out, size=shapes[0], mode="bilinear", align_corners=True)

        if self.training:
            losses += self.criteria(out, x).mean()
            return out, losses
        return out

class VAENorm(nn.Module):   #VAENORM
    def __init__(self, dim, k=2, s=1, p=0, groups=1, reduction=1):
        super(VAENorm, self).__init__()
        self.enc = Encoder(dim, k, s, p, reduction, groups)
        self.dec = Decoder(dim, k, s, p, reduction, groups)

        self.criteria = nn.SmoothL1Loss()

    def forward(self, inputs):
        losses = 0
        shapes = [inputs.shape[2:]]

        inputs = F.interpolate(
            inputs,
            size=(int(shapes[0][0] * 1), int(shapes[0][1] * 1)),
            mode="bilinear",
            align_corners=True,
        )
        inputs_mean, inputs_std = calc_mean_std(inputs)
        # standarlization
        inputs = (inputs - inputs_mean.expand(inputs.size())) / inputs_std.expand(
            inputs.size()
        )
        out = self.dec(self.enc(inputs))
        out = adaptive_instance_normalization_new(out, inputs_mean, inputs_std)
        out = F.interpolate(out, size=shapes[0], mode="bilinear", align_corners=True)
        if self.training:
            losses += self.criteria(out, inputs).mean()
            return out, losses
        return out 

# class VAE(nn.Module):
#     def __init__(self, dim, k=2, s=1, p=0, groups=1, reduction=1):
#         super(VAE, self).__init__()
        
#         self.enc = nn.Sequential(
#             nn.Conv2d(dim, dim, 1, 1),
#             nn.SiLU(),
#             nn.BatchNorm2d(dim),
#             nn.Conv2d(dim, dim // reduction, k, stride=s, padding=p, groups=groups),
#             nn.SiLU(),
#             nn.BatchNorm2d(dim // reduction),
#         )

#         self.dec = nn.Sequential(
#             nn.ConvTranspose2d(
#                 dim // reduction, dim, k, stride=s, padding=p, groups=groups),
#             nn.BatchNorm2d(dim),
#             nn.SiLU(),
#             nn.Conv2d(dim, dim, 1, 1),
#             nn.BatchNorm2d(dim),
#             nn.SiLU(),
#         )

#         self.binarizer = Binarizer(dim // reduction, dim // reduction)

#         self.criteria = nn.SmoothL1Loss()
        
#     def forward(self, x):
#         losses = 0
#         shapes = [x.shape[2:]]

#         out = self.enc(x)
#         out = self.binarizer(out)
#         out = self.dec(out)

#         out = F.interpolate(out, size=shapes[0], mode="bilinear", align_corners=True)
#         if self.training:
#             losses += self.criteria(out, x).mean()
#             return out, losses
#         return out
    
