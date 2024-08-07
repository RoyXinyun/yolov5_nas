import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F
from .vqvae import VectorQuantizer
from .repconv import RepConv


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


class MSLKA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv0_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)

        self.conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)

        self.conv2_1 = nn.Conv2d(dim, dim, (1, 15), padding=(0, 7), groups=dim)
        self.conv2_2 = nn.Conv2d(dim, dim, (15, 1), padding=(7, 0), groups=dim)
        self.conv3 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)
        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)
        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)
        attn = attn + attn_0 + attn_1 + attn_2
        attn = self.conv3(attn)
        return attn * u


class LKA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(
            dim, dim, 5, stride=1, padding=6, groups=dim, dilation=3
        )
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)

        return u * attn


class CA(nn.Module):
    def __init__(self, d_model, ms=False):
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        if ms:
            self.spatial_gating_unit = MSLKA(d_model)
        else:
            self.spatial_gating_unit = LKA(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x


class TinyEncoder(nn.Module):
    def __init__(self, dim, k=2, s=1, p=0, reduction=1, groups=1, ca_type=0):
        super(TinyEncoder, self).__init__()
        if ca_type == 0:
            self.enc = nn.Sequential(
                nn.Conv2d(dim, dim // reduction, k, stride=s, padding=p, groups=groups),
                nn.SiLU(),
                nn.BatchNorm2d(dim // reduction),
            )
        elif ca_type == 1:
            self.enc = nn.Sequential(
                CA(dim, False),
                nn.Conv2d(dim, dim // reduction, k, stride=s, padding=p, groups=groups),
                nn.SiLU(),
                nn.BatchNorm2d(dim // reduction),
                CA(dim // reduction, False),
            )
        elif ca_type == 2:
            self.enc = nn.Sequential(
                nn.Conv2d(dim, dim // reduction, k, stride=s, padding=p, groups=groups),
                nn.SiLU(),
                nn.BatchNorm2d(dim // reduction),
                CA(dim // reduction, True),
            )
        self.binarizer = Binarizer(dim // reduction, dim // reduction)

    def forward(self, x):
        x = self.enc(x)
        x = self.binarizer(x)
        return x


class RepEncoder(nn.Module):
    def __init__(self, dim, k=2, s=1, p=0, reduction=1, groups=1):
        super(RepEncoder, self).__init__()
        self.enc = nn.Sequential(
            RepConv(dim, dim, 3, 1, 1, groups=dim // 4),
            nn.Conv2d(dim, dim, 1, 1),
            RepConv(dim, dim, 3, 1, 1, groups=dim // 2),
            nn.Conv2d(dim, dim, 1, 1),
            RepConv(dim, dim, 3, 1, 1, groups=dim),
            nn.Conv2d(dim, dim // reduction, k, stride=s, padding=p, groups=groups),
            nn.BatchNorm2d(dim // reduction),
            nn.SiLU(),
        )
        self.binarizer = Binarizer(dim // reduction, dim // reduction)

    def forward(self, x):
        x = self.enc(x)
        x = self.binarizer(x)
        return x


class EqRepEncoder(nn.Module):
    def __init__(self, dim, k=2, s=1, p=0, reduction=1, groups=1):
        super(EqRepEncoder, self).__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim // 4),
            nn.Conv2d(dim, dim, 1, 1),
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim // 2),
            nn.Conv2d(dim, dim, 1, 1),
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
            nn.Conv2d(dim, dim // reduction, k, stride=s, padding=p, groups=groups),
            nn.BatchNorm2d(dim // reduction),
            nn.SiLU(),
        )
        self.binarizer = Binarizer(dim // reduction, dim // reduction)

    def forward(self, x):
        x = self.enc(x)
        x = self.binarizer(x)
        return x


class Encoder(nn.Module):
    def __init__(self, dim, k=2, s=1, p=0, reduction=1, groups=1):
        super(Encoder, self).__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1),
            nn.SiLU(),
            nn.BatchNorm2d(dim),
            nn.Conv2d(dim, dim // reduction, k, stride=s, padding=p, groups=groups),
            nn.SiLU(),
            nn.BatchNorm2d(dim // reduction),
        )
        self.binarizer = Binarizer(dim // reduction, dim // reduction)

    def forward(self, x):
        x = self.enc(x)
        x = self.binarizer(x)
        return x


class RepDecoder_1(nn.Module):
    def __init__(self, dim, k=2, s=1, p=0, reduction=1, groups=1):
        super(RepDecoder_1, self).__init__()
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(
                dim // reduction, dim, k, stride=s, padding=p, groups=groups
            ),
            nn.BatchNorm2d(dim),
            nn.SiLU(),
            RepConv(dim, dim, 3, 1, 1, groups=dim),
            nn.Conv2d(dim, dim, 1, 1),
            RepConv(dim, dim, 3, 1, 1, groups=dim // 2),
            nn.Conv2d(dim, dim, 1, 1),
            RepConv(dim, dim, 3, 1, 1, groups=dim // 4),
        )

    def forward(self, x):
        return self.dec(x)


class RepDecoderAddParams(nn.Module):
    def __init__(self, dim, k=2, s=1, p=0, reduction=1, groups=1):
        super(RepDecoderAddParams, self).__init__()
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(
                dim // reduction, dim, k, stride=s, padding=p, groups=groups
            ),
            nn.BatchNorm2d(dim),
            nn.SiLU(),
            RepConv(dim, dim, 3, 1, 1, groups=dim),
            nn.Conv2d(dim, dim, 1, 1),
            RepConv(dim, dim, 3, 1, 1, groups=dim // 2),
            nn.Conv2d(dim, dim, 1, 1),
            RepConv(dim, dim, 3, 1, 1, groups=dim // 4),
            nn.Conv2d(dim, dim, 1, 1),
            RepConv(dim, dim, 3, 1, 1, groups=1),
            nn.Conv2d(dim, dim, 1, 1),
            RepConv(dim, dim, 3, 1, 1, groups=1),
        )

    def forward(self, x):
        return self.dec(x)


class EqRepDecoder(nn.Module):
    def __init__(self, dim, k=2, s=1, p=0, reduction=1, groups=1):
        super(EqRepDecoder, self).__init__()
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(
                dim // reduction, dim, k, stride=s, padding=p, groups=groups
            ),
            nn.BatchNorm2d(dim),
            nn.SiLU(),
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
            nn.Conv2d(dim, dim, 1, 1),
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim // 2),
            nn.Conv2d(dim, dim, 1, 1),
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim // 4),
            nn.Conv2d(dim, dim, 1, 1),
            nn.Conv2d(dim, dim, 3, 1, 1, groups=1),
        )

    def forward(self, x):
        return self.dec(x)


class EqRepDecoderSym(nn.Module):
    def __init__(self, dim, k=2, s=1, p=0, reduction=1, groups=1):
        super(EqRepDecoderSym, self).__init__()
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(
                dim // reduction, dim, k, stride=s, padding=p, groups=groups
            ),
            nn.BatchNorm2d(dim),
            nn.SiLU(),
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
            nn.Conv2d(dim, dim, 1, 1),
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim // 2),
            nn.Conv2d(dim, dim, 1, 1),
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim // 4),
        )

    def forward(self, x):
        return self.dec(x)


class RepDecoder(nn.Module):
    def __init__(self, dim, k=2, s=1, p=0, reduction=1, groups=1):
        super(RepDecoder, self).__init__()
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(
                dim // reduction, dim, k, stride=s, padding=p, groups=groups
            ),
            nn.BatchNorm2d(dim),
            nn.SiLU(),
            RepConv(dim, dim, 3, 1, 1, groups=dim),
            nn.Conv2d(dim, dim, 1, 1),
            RepConv(dim, dim, 3, 1, 1, groups=dim // 2),
            nn.Conv2d(dim, dim, 1, 1),
            RepConv(dim, dim, 3, 1, 1, groups=dim // 4),
            nn.Conv2d(dim, dim, 1, 1),
            RepConv(dim, dim, 3, 1, 1, groups=1),
        )

    def forward(self, x):
        return self.dec(x)


class Decoder(nn.Module):
    def __init__(self, dim, k=2, s=1, p=0, reduction=1, groups=1):
        super(Decoder, self).__init__()
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(
                dim // reduction, dim, k, stride=s, padding=p, groups=groups
            ),
            nn.BatchNorm2d(dim),
            nn.SiLU(),
            nn.Conv2d(dim, dim, 1, 1),
            nn.BatchNorm2d(dim),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.dec(x)


class LKADecoder(nn.Module):
    def __init__(self, dim, k=2, s=1, p=0, reduction=1, ms=False):
        super(LKADecoder, self).__init__()
        self.dec = nn.Sequential(
            CA(dim // reduction, ms),
            nn.Conv2d(dim // reduction, dim, 1, 1),
            nn.BatchNorm2d(dim),
            nn.SiLU(),
            nn.ConvTranspose2d(dim, dim, k, stride=s, padding=p),
            nn.BatchNorm2d(dim),
            nn.SiLU(),
            CA(dim, ms),
        )

    def forward(self, x):
        return self.dec(x)


class VQVAEOneBranch1(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=0, g=1, r=1, loss_type=0):
        super(VQVAEOneBranch1, self).__init__()
        self.enc_vq = nn.Sequential(
            nn.Conv2d(c1, c1, 3, 1, 1, groups=c1),
            nn.SiLU(),
            nn.BatchNorm2d(c1),
            nn.Conv2d(c1, c1 // 2, 1, 1),
            nn.SiLU(),
            nn.BatchNorm2d(c1 // 2),
            nn.Conv2d(c1 // 2, c1, 1, 1),
            nn.SiLU(),
            nn.BatchNorm2d(c1),
            nn.Conv2d(c1, c1, 3, 1, 1, groups=c1),
            nn.SiLU(),
            nn.BatchNorm2d(c1),
        )
        self.enc = nn.Sequential(
            nn.Conv2d(c1, c1 // r, k, stride=s, padding=p, groups=g),
            nn.SiLU(),
            nn.BatchNorm2d(c1 // r),
        )
        self.vq = VectorQuantizer()
        self.binarizer = Binarizer(c1 // r, c1 // r)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(c1 // r, c1, k, stride=s, padding=p, groups=g),
            nn.BatchNorm2d(c1),
            nn.SiLU(),
        )
        self.dec_vq = nn.Sequential(
            nn.Conv2d(c1, c1 // 2, 3, 1, 1),
            nn.BatchNorm2d(c1 // 2),
            nn.SiLU(),
            nn.Conv2d(c1 // 2, c1, 1, 1),
            nn.BatchNorm2d(c1),
            nn.SiLU(),
            nn.Conv2d(c1, c1, 3, 1, 1),
            nn.BatchNorm2d(c1),
            nn.SiLU(),
        )

        self.criteria = nn.SmoothL1Loss()
        self.loss_type = loss_type

    def forward(self, inputs):
        losses = 0
        shapes = [inputs.shape[2:]]
        inputs = self.enc_vq(inputs)
        enc_feat = self.enc(inputs)
        dec_feat = self.dec(self.binarizer(enc_feat))
        enc_feat_q, loss_vq, _ = self.vq(inputs)
        dec_feat = F.interpolate(
            dec_feat, size=shapes[0], mode="bilinear", align_corners=True
        )
        out = self.dec_vq(dec_feat + enc_feat_q)

        if self.training:
            if self.loss_type == 0:
                losses += self.criteria(out, inputs).mean()
            losses += loss_vq.mean()
            return out, losses
        return out


class VAEOneBranch(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=0, g=1, r=1, decoder_type=0, encoder_type=0):
        super(VAEOneBranch, self).__init__()
        if encoder_type == 0:
            self.enc0 = Encoder(c1, k, s, p, r, g)
        elif encoder_type == 1:
            self.enc0 = TinyEncoder(c1, k, s, p, r, g)
        elif encoder_type == 2:
            self.enc0 = TinyEncoder(c1, k, s, p, r, g, ca_type=1)
        elif encoder_type == 3:
            self.enc0 = TinyEncoder(c1, k, s, p, r, g, ca_type=2)
        elif encoder_type == 4:
            self.enc0 = RepEncoder(c1, k, s, p, r)
        elif encoder_type == 5:
            pass
        elif encoder_type == 6:
            self.enc0 = EqRepEncoder(c1, k, s, p, r)
        if decoder_type == 0:
            self.dec0 = Decoder(c1, k, s, p, r, g)
        elif decoder_type == 1:
            self.dec0 = LKADecoder(c1, k, s, p, r)
        elif decoder_type == 2:
            self.dec0 = LKADecoder(c1, k, s, p, r, ms=True)
        elif decoder_type == 3:
            self.dec0 = RepDecoder(c1, k, s, p, r, g)
        elif decoder_type == 4:
            self.dec0 = RepDecoder_1(c1, k, s, p, r, g)
        elif decoder_type == 5:
            self.dec0 = RepDecoderAddParams(c1, k, s, p, r, g)
        elif decoder_type == 6:
            self.dec0 = EqRepDecoder(c1, k, s, p, r, g)
        elif decoder_type == 7:
            self.dec0 = EqRepDecoderSym(c1, k, s, p, r, g)
        self.criteria = nn.SmoothL1Loss()

    def forward(self, inputs):
        losses = 0
        shapes = [inputs.shape[2:]]
        out = self.dec0(self.enc0(inputs))

        out = F.interpolate(out, size=shapes[0], mode="bilinear", align_corners=True)
        if self.training:
            losses += self.criteria(out, inputs).mean()
            return out, losses
        return out


class VAEOneBranchNorm(VAEOneBranch):
    def __init__(self, c1, c2, k=1, s=1, p=0, g=1, r=1, decoder_type=0, encoder_type=0):
        super(VAEOneBranchNorm, self).__init__(
            c1, c2, k, s, p, g, r, decoder_type, encoder_type
        )

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
        inputs = (inputs - inputs_mean.expand(inputs.size())) / inputs_std.expand(
            inputs.size()
        )
        out = self.dec0(self.enc0(inputs))
        out = adaptive_instance_normalization_new(out, inputs_mean, inputs_std)
        out = F.interpolate(out, size=shapes[0], mode="bilinear", align_corners=True)
        if self.training:
            losses += self.criteria(out, inputs).mean()
            return out, losses
        return out


class VAEOneBranchNormInfer(VAEOneBranchNorm):
    def __init__(self, c1, c2, k=1, s=1, p=0, g=1, r=1, decoder_type=0, encoder_type=0):
        super(VAEOneBranchNormInfer, self).__init__(
            c1, c2, k, s, p, g, r, decoder_type, encoder_type
        )
        self.encoder_type = encoder_type
        self.decoder_type = decoder_type

    def forward(self, inputs):
        if self.decoder_type == -1:
            inputs_mean, inputs_std = calc_mean_std(inputs)
            shapes = inputs.shape[2:]
            inputs = (inputs - inputs_mean.expand(inputs.size())) / inputs_std.expand(
                inputs.size()
            )
            out = self.enc0(inputs)

            return {
                "outs": out,
                "shapes": shapes,
                "inputs_mean": inputs_mean,
                "inputs_std": inputs_std,
            }
        elif self.encoder_type == -1:
            outs = inputs["outs"]
            shapes = inputs["shapes"]
            inputs_mean = inputs["inputs_mean"]
            inputs_std = inputs["inputs_std"]
            outs = self.dec0(outs)
            outs = adaptive_instance_normalization_new(outs, inputs_mean, inputs_std)

            outs = F.interpolate(outs, size=shapes, mode="bilinear", align_corners=True)
            return outs


class VAEOneBranchInfer(VAEOneBranch):
    def __init__(self, c1, c2, k=1, s=1, p=0, g=1, r=1, decoder_type=0, encoder_type=0):
        super(VAEOneBranchInfer, self).__init__(
            c1, c2, k, s, p, g, r, decoder_type, encoder_type
        )
        self.encoder_type = encoder_type
        self.decoder_type = decoder_type

    def forward(self, inputs):
        if self.decoder_type == -1:
            shapes = inputs.shape[2:]
            out = self.enc0(inputs)

            return {"outs": out, "shapes": shapes}
        elif self.encoder_type == -1:
            outs = inputs["outs"]
            shapes = inputs["shapes"]
            outs = self.dec0(outs)

            outs = F.interpolate(outs, size=shapes, mode="bilinear", align_corners=True)
            return outs
