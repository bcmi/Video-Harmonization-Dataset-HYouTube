import torch
from functools import partial

from torch import nn as nn
import torch.nn.functional as F
import  numpy as np

from iharm.model.base.rain.util.config import cfg
from iharm.model.base.rain.models.networks import RainNet
from iharm.model.base.rain.models.normalize import RAIN
from iharm.model.base.rain.util import util as rainutil
from iharm.model.modeling.basic_blocks import ConvBlock, GaussianSmoothing
from iharm.model.modeling.unet import UNetEncoder, UNetDecoder
from iharm.model.ops import ChannelAttention
from iharm.model.modeling.dbp import NewRes, SimpleRefine
import time
from iharm.utils.misc import load_weights
from iharm.model.base import SSAMImageHarmonization
from iharm.model.modeling.lut import TrilinearInterpolation, TridistributeGeneraotr


class SSAMvideoLut(nn.Module):
    def __init__(
            self,
            depth, device = None, backbone_path = "" , with_lutoutput = False, use_feature=True, need_normalize = True, need_denormalize = True,
            backbone_type = 'issam'
    ):
        super(SSAMvideoLut, self).__init__()
        self.backbone_type = backbone_type
        if self.backbone_type == 'issam':
            self.mean = torch.tensor([.485, .456, .406], dtype=torch.float32).view(1, 3, 1, 1)
            self.std = torch.tensor([.229, .224, .225], dtype=torch.float32).view(1, 3, 1, 1)
        elif self.backbone_type == 'rain':
            self.mean = torch.tensor([.5, .5, .5], dtype=torch.float32).view(1, 3, 1, 1)
            self.std = torch.tensor([.5, .5, .5], dtype=torch.float32).view(1, 3, 1, 1)
        self.depth = depth

        self.use_feature = use_feature
        self.need_normalize = need_normalize
        self.need_denormalize = need_denormalize
        self.backbone_checkpoint = backbone_path
        # issam
        self.with_lutoutput = with_lutoutput
        self.device = device
        if self.backbone_type == 'issam':
            self.backbone = SSAMImageHarmonization(
            depth=self.depth, ch=32, image_fusion=True, attention_mid_k=0.5,
            attend_from=2, batchnorm_from=2)
            load_weights(self.backbone, self.backbone_checkpoint)

        elif self.backbone_type == 'rain':
            self.backbone = RainNet(input_nc=cfg.input_nc,
                                    output_nc=cfg.output_nc,
                                    ngf=cfg.ngf,
                                    norm_layer=RAIN,
                                    use_dropout=not cfg.no_dropout)
            state_dict = torch.load(self.backbone_checkpoint)
            rainutil.copy_state_dict(self.backbone.state_dict(), state_dict)
            print("load backbone from {}".format(self.backbone_checkpoint))



        #
        self.lut_dim = 33
        self.lut_generator = TridistributeGeneraotr()
        self.trilinear = TrilinearInterpolation()
        # dbp refinement
        if self.backbone_type == 'issam':
            in_channel = 32 if use_feature else 0
            self.refine_network = SimpleRefine(feature_channels=in_channel, inner_channel=64)
        elif self.backbone_type == 'rain':
            in_channel = 128 if use_feature else 0
            self.refine_network = SimpleRefine(feature_channels=in_channel, inner_channel=64)

    def load_backbone(self):
        if self.backbone_type == 'issam':
            load_weights(self.backbone, self.backbone_checkpoint)
        elif self.backbone_type == 'rain':
            state_dict = torch.load(self.backbone_checkpoint)
            rainutil.copy_state_dict(self.backbone.state_dict(), state_dict)
        print("load backbone")
        #exit()
    def init_device(self, input_device):
        if self.device is None:
            self.device = input_device
            self.mean = self.mean.to(self.device)
            self.std = self.std.to(self.device)

    def normalize(self, tensor):
        self.init_device(tensor.device)
        # return self.norm(tensor)
        return (tensor - self.mean) / self.std

    def denormalize(self, tensor):
        self.init_device(tensor.device)
        return tensor * self.std + self.mean

    def mask_denormalize(self, tensor, mask, original):
        self.init_device(tensor.device)
        tmp_res = tensor * self.std + self.mean
        return tmp_res*mask + original * (1-mask)


    def train(self, mode=True):
        if self.backbone_type == 'issam':
            self.backbone.eval()
        elif self.backbone_type == 'rain':
            self.backbone.eval()
        self.refine_network.train()

    def eval(self):
        if self.backbone_type == 'issam':
            self.backbone.eval()
        elif self.backbone_type == 'rain':
            self.backbone.eval()
        self.refine_network.eval()

    def forward(self, image, mask, backbone_features=None, previous={}, direct_lutoutput = None, direct_lut_map = None, names=[]):
        dim = 33
        #print(self.need_normalize, self.need_denormalize)
        #exit()
        if not self.with_lutoutput:
            previous_images = previous['images']
            previous_masks = previous['masks']
        if self.need_normalize:
            normaled_images = self.normalize(image)
        else:
            normaled_images = image
        with torch.no_grad():
            if self.backbone_type == 'issam':
                cur_backbone_result = self.backbone(normaled_images, mask)
                cur_backbone_output, cur_backbone_feature = cur_backbone_result['images'], cur_backbone_result['features']
            elif self.backbone_type == 'rain':
                cur_backbone_feature, cur_backbone_output = self.backbone.processImage(normaled_images, mask)
        if self.need_denormalize:
            denormaled_cur_backbone_output = self.mask_denormalize(cur_backbone_output, mask, image)
        else:
            denormaled_cur_backbone_output = cur_backbone_output
        if not self.with_lutoutput:
            previous_num = previous_images.shape[1]
            pre_backbone_outputs = []
            for index in range(previous_num):
                if self.need_normalize:
                    normaled_previous_images = self.normalize(previous_images[:, index, :, :, :])
                else:
                    normaled_previous_images = previous_images[:, index, :, :, :]
                with torch.no_grad():
                    if self.backbone_type == 'issam':
                        pre_backbone_result = self.backbone(normaled_previous_images, previous_masks[:, index, :, :, :])
                        normaled_pre_backbone_output = pre_backbone_result['images']
                    elif self.backbone_type == 'rain':
                        _, normaled_pre_backbone_output = self.backbone.processImage(normaled_previous_images,
                                                                                     previous_masks[:, index, :, :, :])
                    pre_backbone_outputs.append(
                        self.mask_denormalize(normaled_pre_backbone_output, previous_masks[:, index, :, :, :],
                                              previous_images[:, index, :, :, :]))
            pre_backbone_outputs = torch.stack(pre_backbone_outputs, dim=1)

            # luts = torch.zeros((3,33,33,33)).to(normaled_pre_issam_output.device)
            # lut_counts = torch.zeros((3,33,33,33)).to(normaled_pre_issam_output.device)
            # pre issam outputs seq*batch*3*256*256, previous image batch * seq*3*256*256

            batch = image.shape[0]
            w = previous_masks.shape[-2]
            h = previous_masks.shape[-1]

            tmp_previous_masks = torch.reshape(previous_masks.permute((0, 2, 1, 3, 4)),
                                               (batch, 1, w * previous_num, h))
            tmp_previous_images = torch.reshape(previous_images.permute((0, 2, 1, 3, 4)),
                                                (batch, 3, w * previous_num, h))
            tmp_pre_backbone_outputs = torch.reshape(pre_backbone_outputs.permute((0, 2, 1, 3, 4)),
                                                  (batch, 3, w * previous_num, h))

            luts, lut_counts, _ = self.lut_generator(tmp_previous_masks,
                                                     tmp_previous_images,
                                                     tmp_pre_backbone_outputs)

            #luts, lut_counts = self.lut_generator.divide(luts, lut_counts)
            _, lut_output = self.trilinear(lut_counts, luts, image)
            lut_output = lut_output * mask + image * (1 - mask)

            lut_map = self.trilinear.count_map(lut_counts, image)
            lut_map = lut_map * mask
            lut_output = lut_output * (1 - lut_map) + lut_map * denormaled_cur_backbone_output

        else:
            lut_output = direct_lutoutput
            lut_output = lut_output * (1 - direct_lut_map) + direct_lut_map * denormaled_cur_backbone_output
        tmp_lut_output = lut_output
        #lut_output = self.normalize(lut_output)
        if self.need_normalize:
            lut_output = self.normalize(lut_output)
        t3 = time.time()
        final_output = self.refine_network(cur_backbone_output, lut_output,  image, cur_backbone_feature)
        t4 = time.time()
        if self.need_denormalize:
            denormaled_final_output = self.mask_denormalize(final_output, mask, image)
        else:
            denormaled_final_output = final_output
        #print((denormaled_final_output - lut_output).sum())
        return {"images":denormaled_final_output, "backbone_out":denormaled_cur_backbone_output, "lut_output":tmp_lut_output}

