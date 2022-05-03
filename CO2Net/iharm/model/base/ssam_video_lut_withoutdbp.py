import torch
from functools import partial

from torch import nn as nn
import torch.nn.functional as F
import  numpy as np
import cv2
import os
import copy
from skimage.measure import compare_mse as mse
from iharm.model.base.rain.util.config import cfg
from iharm.model.base.rain.models.networks import RainNet
from iharm.model.base.rain.models.normalize import RAIN
from torchvision import transforms
from iharm.model.modeling.basic_blocks import ConvBlock, GaussianSmoothing
from iharm.model.modeling.unet import UNetEncoder, UNetDecoder
from iharm.model.ops import ChannelAttention
from iharm.model.modeling.dbp import NewRes, SimpleRefine
import time
from iharm.model.base.rain.util import util as rainutil
from iharm.utils.misc import load_weights
from iharm.model.base import SSAMImageHarmonization
from iharm.inference.transforms import NormalizeTensor, PadToDivisor, ToTensor, AddFlippedTensor
from iharm.model.modeling.lut import TrilinearInterpolation, TridistributeGeneraotr



write_dir = '/home/ubuntu/tensors/2_2/'


class SSAMvideoLutWithoutDbp(nn.Module):
    def __init__(
            self,
            depth, device = None, backbone_path = "", use_lutoutput = False, fix_threshold = 0.1, k_threshold = 1,
            write_lut_output = '', write_lut_map="", backbone_type = 'issam'
    ):
        super(SSAMvideoLutWithoutDbp, self).__init__()
        self.use_lutoutput = use_lutoutput
        self.backbone_type = backbone_type
        if self.backbone_type == 'issam':
            self.mean = torch.tensor([.485, .456, .406], dtype=torch.float32).view(1, 3, 1, 1)
            self.std = torch.tensor([.229, .224, .225], dtype=torch.float32).view(1, 3, 1, 1)
        elif self.backbone_type == 'rain':
            self.mean = torch.tensor([.5, .5, .5], dtype=torch.float32).view(1, 3, 1, 1)
            self.std = torch.tensor([.5, .5, .5], dtype=torch.float32).view(1, 3, 1, 1)
        self.depth = depth
        self.write_lut_output = write_lut_output
        self.write_lut_map = write_lut_map
        self.backbone_checkpoint = backbone_path
        # issam
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



        #
        self.lut_dim = 33
        self.lut_generator = TridistributeGeneraotr()
        self.trilinear = TrilinearInterpolation(fix_threshold, k_threshold)
        # dbp refinement
        #self.refine_network = SimpleRefine(feature_channels=32, inner_channel=64)

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


    def train(self, mode=True):
        self.backbone.eval()
        #self.refine_network.train()

    def eval(self):
        self.backbone.eval()
        #self.refine_network.eval()

    def forward(self, image, mask, backbone_features=None, previous={}, names = [], direct_lutoutput=None, direct_lut_map=None):
        #print(previous)
        torch.cuda.synchronize()
        t0 = time.time()
        get_lut_time = 0
        divide_time = 0
        tri_linear_time = 0
        previous_images = previous['images']
        previous_masks = previous['masks']
        #print(previous_images.shape)
        torch.cuda.synchronize()
        t1 = time.time()
        ##issam step
        normaled_images = self.normalize(image)
        with torch.no_grad():
            if self.backbone_type == 'issam':
                cur_backbone_result = self.backbone(normaled_images, mask)
                cur_backbone_output, cur_backbone_feature = cur_backbone_result['images'], cur_backbone_result['features']
            elif self.backbone_type == 'rain':
                cur_backbone_feature, cur_backbone_output = self.backbone.processImage(normaled_images, mask)
        denormaled_cur_backbone_output = self.mask_denormalize(cur_backbone_output, mask, image)

        if not self.use_lutoutput:
            previous_num = previous_images.shape[1]
            pre_backbone_outputs = []
            for index in range(previous_num):
                normaled_previous_images = self.normalize(previous_images[:, index, :, :, :])
                with torch.no_grad():
                    if self.backbone_type == 'issam':
                        pre_backbone_result = self.backbone(normaled_previous_images, previous_masks[:, index, :, :, :])
                        normaled_pre_backbone_output = pre_backbone_result['images']
                    elif self.backbone_type == 'rain':
                        normaled_pre_backbone_feature, normaled_pre_backbone_output = self.backbone.processImage(normaled_previous_images, previous_masks[:, index, :, :, :])
                    pre_backbone_outputs.append(self.mask_denormalize(normaled_pre_backbone_output, previous_masks[:, index, :, :, :], previous_images[:, index, :, :, :]))

            pre_backbone_outputs = torch.stack(pre_backbone_outputs, dim =1)


            batch = image.shape[0]
            w = previous_masks.shape[-2]
            h = previous_masks.shape[-1]
            torch.cuda.synchronize()
            tmp_previous_masks = torch.reshape(previous_masks.permute((0, 2, 1, 3, 4)),
                                               (batch, 1, w * previous_num, h))
            tmp_previous_images = torch.reshape(previous_images.permute((0, 2, 1, 3, 4)),
                                                (batch, 3, w * previous_num, h))
            tmp_pre_backbone_outputs = torch.reshape(pre_backbone_outputs.permute((0, 2, 1, 3, 4)),
                                                     (batch, 3, w * previous_num, h))

            luts, lut_counts, _ = self.lut_generator(tmp_previous_masks,
                                                   tmp_previous_images,
                                                   tmp_pre_backbone_outputs)
            _, lut_output = self.trilinear(lut_counts, luts, image)
            lut_output = lut_output * mask + image * (1 - mask)

            lut_map = self.trilinear.count_map(lut_counts, image)
            lut_map = lut_map * mask

            ###

            _, pre_lut_output = self.trilinear(lut_counts, luts, tmp_previous_images)
            pre_lut_map =  self.trilinear.count_map(lut_counts, tmp_previous_images)
            pre_lut_map = pre_lut_map * tmp_previous_masks
            pre_lut_output = (1 - pre_lut_map) * pre_lut_output + pre_lut_map * tmp_pre_backbone_outputs
            
            pre_lut_output = pre_lut_output * tmp_previous_masks + (1 - tmp_previous_masks) * tmp_previous_images
            tmp_pre_backbone_outputs = tmp_pre_backbone_outputs * tmp_previous_masks + (1 - tmp_previous_masks) * tmp_pre_backbone_outputs

            invalid_rate = 0
            for b in range(batch):
                lut_m = lut_map[b].detach().cpu().numpy()
                _, lw, lh = lut_m.shape
                invalid_rate += lut_m.sum() / lw / lh
            
            total_me = 0
            for b in range(batch):
                tmp_fore = tmp_previous_masks[b].detach().cpu().numpy().sum()
                pre_lut_output_single = torch.clamp(pre_lut_output[b] * 255, 0, 255)
                tmp_pre_backbone_output = torch.clamp(tmp_pre_backbone_outputs[b] * 255, 0, 255)
                _, w, h = pre_lut_output_single.shape
                total_me += mse(pre_lut_output_single.detach().cpu().numpy(), tmp_pre_backbone_output.detach().cpu().numpy()) * w * h / tmp_fore


            lut_output = lut_output * (1 - lut_map) + lut_map * denormaled_cur_backbone_output

        else:
            lut_output = direct_lutoutput
            lut_map = direct_lut_map
        batch = image.shape[0]
        if len(self.write_lut_output) > 0:
            for b in range(batch):
                video, obj, img_num = names[b].split('/')[-3:]
                new_name = video + '_' + obj + '_' + img_num[:-3] + 'npy'
                np.save(os.path.join(self.write_lut_output, new_name), lut_output[b].detach().cpu().numpy())
                np.save(os.path.join(self.write_lut_map, new_name), lut_map[b].detach().cpu().numpy())


        lut_output = lut_output * (1-lut_map) + lut_map *denormaled_cur_backbone_output
        lut_output = lut_output * mask + image * (1 - mask)


        return {"images":lut_output, "backbone_out":denormaled_cur_backbone_output, "lut_output":lut_output, "me":total_me, "invalid":invalid_rate}


