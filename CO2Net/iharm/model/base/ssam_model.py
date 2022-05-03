import torch
from functools import partial

from torch import nn as nn
import time
from iharm.model.modeling.basic_blocks import ConvBlock, GaussianSmoothing
from iharm.model.modeling.unet import UNetEncoder, UNetDecoder
from iharm.model.ops import ChannelAttention


class SSAMImageHarmonization(nn.Module):
    def __init__(
        self,
        depth,
        norm_layer=nn.BatchNorm2d, batchnorm_from=2,
        attend_from=3, attention_mid_k=2.0,
        need_normalize = False,
        image_fusion=True,
        ch=64, max_channels=512,
        backbone_from=-1, backbone_channels=None, backbone_mode=''
    ):
        super(SSAMImageHarmonization, self).__init__()
        self.depth = depth
        self.device = None
        self.need_normalize = need_normalize
        self.mean = torch.tensor([.485, .456, .406], dtype=torch.float32).view(1, 3, 1, 1)
        self.std = torch.tensor([.229, .224, .225], dtype=torch.float32).view(1, 3, 1, 1)
        self.encoder = UNetEncoder(
            depth, ch,
            norm_layer, batchnorm_from, max_channels,
            backbone_from, backbone_channels, backbone_mode
        )
        self.decoder = UNetDecoder(
            depth, self.encoder.block_channels,
            norm_layer,
            attention_layer=partial(SpatialSeparatedAttention, mid_k=attention_mid_k),
            attend_from=attend_from,
            image_fusion=image_fusion
        )

    def init_device(self, input_device):
        if self.device is None:
            self.device = input_device
            self.mean = self.mean.to(self.device)
            self.std = self.std.to(self.device)

    def mask_denormalize(self, tensor, mask, original):
        self.init_device(tensor.device)
        tmp_res = tensor * self.std + self.mean
        return tmp_res*mask + original * (1-mask)


    def normalize(self, tensor):
        self.init_device(tensor.device)
        return (tensor - self.mean) / self.std

    def denormalize(self, tensor):
        self.init_device(tensor.device)
        return tensor * self.std + self.mean

    def forward(self, image, mask, backbone_features=None):
        if self.need_normalize:
            image = self.normalize(image)
        x = torch.cat((image, mask), dim=1)
        intermediates = self.encoder(x, backbone_features)
        output, output_map = self.decoder(intermediates, image, mask)
        if self.need_normalize:
            output = self.mask_denormalize(output, mask, image)
        #print("net cost:", start.elapsed_time(end))
        return {"images":output, "features":output_map}


class SpatialSeparatedAttention(nn.Module):
    def __init__(self, in_channels, norm_layer, activation, mid_k=2.0):
        super(SpatialSeparatedAttention, self).__init__()
        self.background_gate = ChannelAttention(in_channels)
        self.foreground_gate = ChannelAttention(in_channels)
        self.mix_gate = ChannelAttention(in_channels)

        mid_channels = int(mid_k * in_channels)
        self.learning_block = nn.Sequential(
            ConvBlock(
                in_channels, mid_channels,
                kernel_size=3, stride=1, padding=1,
                norm_layer=norm_layer, activation=activation,
                bias=False,
            ),
            ConvBlock(
                mid_channels, in_channels,
                kernel_size=3, stride=1, padding=1,
                norm_layer=norm_layer, activation=activation,
                bias=False,
            ),
        )
        self.mask_blurring = GaussianSmoothing(1, 7, 1, padding=3)

    def forward(self, x, mask):
        mask = self.mask_blurring(nn.functional.interpolate(
            mask, size=x.size()[-2:],
            mode='bilinear', align_corners=True
        ))
        background = self.background_gate(x)
        foreground = self.learning_block(self.foreground_gate(x))
        mix = self.mix_gate(x)
        output = mask * (foreground + mix) + (1 - mask) * background
        return output
