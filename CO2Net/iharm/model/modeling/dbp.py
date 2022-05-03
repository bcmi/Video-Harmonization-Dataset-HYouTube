import torch
from torch import nn as nn

from iharm.model.modeling.basic_blocks import ConvBlock, DBDownsample, DBUpsample, UpBlock, DownBlock
import torch.nn.functional as F


class SimpleRefine(nn.Module):
    def __init__(self, feature_channels = 0, in_channel = 6, inner_channel = 32,
                norm_layer = nn.BatchNorm2d, activation = nn.ELU, image_fusion = True):
        super(SimpleRefine, self).__init__()
        self.image_fusion = image_fusion
        self.in_channel = in_channel
        self.feature_channels = feature_channels
        self.refine_block = nn.Sequential(
            nn.Conv2d(feature_channels + in_channel, inner_channel, kernel_size=3, stride=1, padding=1),
            norm_layer(inner_channel) if norm_layer is not None else nn.Identity(),
            activation(),
            nn.Conv2d(inner_channel, inner_channel, kernel_size=3, stride=1, padding=1),
            norm_layer(inner_channel) if norm_layer is not None else nn.Identity(),
            activation(),
        )
        if self.image_fusion:
            self.conv_attention = nn.Conv2d(inner_channel, 1, kernel_size=1)
        self.to_rgb = nn.Conv2d(inner_channel, 3, 1,1,0)

    def forward(self, ssam_out, lut_out, comp, ssam_feature=None):
        if self.feature_channels > 0:
            input = torch.cat([ssam_out, ssam_feature, lut_out], dim=1)
        else:
            input = torch.cat([ssam_out, lut_out], dim=1)
        output_map = self.refine_block(input)
        if self.image_fusion:
            attention_map = torch.sigmoid(3.0 * self.conv_attention(output_map))
            output = attention_map * comp + (1.0 - attention_map) * self.to_rgb(output_map)
        else:
            output = self.to_rgb(output_map)

        return output


class NewRes(nn.Module):
    def __init__(self,feature_channels = 32,in_channel = 7,inner_channel=32,
                 norm_layer=nn.BatchNorm2d, activation=nn.ELU,image_fusion=True):
        super(NewRes, self).__init__()
        self.image_fusion = image_fusion
        self.block = nn.Sequential(
            nn.Conv2d(feature_channels + in_channel, inner_channel, kernel_size=3, stride=1, padding=1),
            norm_layer(inner_channel) if norm_layer is not None else nn.Identity(),
            activation(),
            nn.Conv2d(inner_channel, inner_channel, kernel_size=3, stride=1, padding=1),
            norm_layer(inner_channel) if norm_layer is not None else nn.Identity(),
            activation(),
        )
        if self.image_fusion:
            self.conv_attention = nn.Conv2d(inner_channel, 1, kernel_size=1)
        self.to_rgb = nn.Conv2d(inner_channel, 3, 1,1,0)

    def forward(self, ssam_output, comp, mask, ssam_features,lut_output, target_resolution):
        ssam_in = F.interpolate(ssam_output, size=target_resolution, mode='bilinear')
        comp = F.interpolate(comp, size=target_resolution, mode='bilinear')
        mask = F.interpolate(mask, size=target_resolution, mode='bilinear')
        ssam_features = F.interpolate(ssam_features, size=target_resolution, mode='bilinear')
        lut_in = F.interpolate(lut_output, size=target_resolution, mode='bilinear')
        input_1 = torch.cat([ssam_in, lut_in, mask, ssam_features], dim=1)
        output_map = self.block(input_1)

        if self.image_fusion:
            attention_map = torch.sigmoid(3.0 * self.conv_attention(output_map))
            output = attention_map * comp + (1.0 - attention_map) * self.to_rgb(output_map)
        else:
            output = self.to_rgb(output_map)
        return output_map,output


class DBPNetv1(nn.Module):
    def __init__(self, in_channels=7, feat_in_channels=32, minf=24, depth=2, image_fusion=False):
        super(DBPNetv1, self).__init__()
        self.hdconv_blocks = nn.ModuleList()
        self.hddeconv_blocks = nn.ModuleList()
        # depth = 2
        nhdf = 32 #64
        # minf = 24 # 32
        # in_channels = 7
        # feat_in_channels = 32 
        self.upsample = nn.Upsample(scale_factor=2)
        self.depth = depth
        # self.mode = hd_mode
        self.image_fusion = image_fusion
        out_chs = []
        out_channels = minf
        # self.in_conv = Conv2dBlock(in_channels, nhdf, 3,1,1, norm='none', activation='elu')
        self.in_conv = nn.Sequential(*[
            ConvBlock(in_channels, out_channels, kernel_size=3, stride=1, padding=1, norm_layer=None),
            nn.Conv2d(out_channels, out_channels, 3,1,1),
        ])
        self.feat_conv = nn.Sequential(*[
            ConvBlock(feat_in_channels, out_channels, kernel_size=3, stride=1, padding=1, norm_layer=None),
            nn.Conv2d(out_channels, out_channels, 3,1,1),
        ])

        in_channels = nhdf
        for d in range(depth):
            out_channels = min(int(nhdf * (2**(d))), minf)
            out_chs.append(out_channels)
            if d != depth-1:
                self.hdconv_blocks.append(nn.Sequential(*[
                    DBDownsample(out_channels, out_channels, ks=4, activation='elu', norm='sn', activation_first=False),
                    DBUpsample(out_channels, out_channels, ks=4, activation='elu', norm='sn',activation_first=False)
                ]))
            else:
                self.hdconv_blocks.append(nn.Sequential(*[
                    DBDownsample(out_channels, out_channels, ks=4, activation='elu', norm='sn',activation_first=False),
                    DBUpsample(out_channels, out_channels, ks=4, activation='elu', norm='sn',activation_first=False)
                ]))
            in_channels = out_channels
        for d in range(depth-1):
            self.hddeconv_blocks.append(nn.Sequential(*[
                    DBDownsample(out_chs[d], out_chs[d], ks=4, activation='elu', norm='sn',activation_first=False),
                    DBUpsample(out_chs[d], out_chs[d], ks=4, activation='elu', norm='sn',activation_first=False)
            ]))
            
        out_channels = out_chs[0]
        
        if self.image_fusion:
            self.conv_attention = nn.Conv2d(out_channels, 1, kernel_size=1)
        self.to_rgb = nn.Conv2d(out_channels, 3, 1,1,0)
    
    def forward(self, lr_output, hr_mask, hr_comp, extra_feat=None, lut_hr_output=None, target_resolution=(1024,1024)):
        up_output = F.interpolate(lr_output, size=target_resolution, mode='bilinear')
        extra_feat = F.interpolate(extra_feat, size=target_resolution, mode='bilinear') # C=32
        im_all = torch.cat([up_output,lut_hr_output,hr_mask],dim=1)
        im_all = im_all.contiguous()
        hx = self.in_conv(im_all) + self.feat_conv(extra_feat)
        skips = []
        # print(len(self.hdconv_blocks),len(self.hddeconv_blocks))
        for block in self.hdconv_blocks:
            hx = block(hx)
            skips.append(hx)
            # print(hx.shape)
        prev_out = skips.pop()
        for block in self.hddeconv_blocks[::-1]:
            prev_out = F.interpolate(prev_out, size=skips[-1].shape[2:], mode='bilinear')
            skip = skips.pop()
            block_in = prev_out + skip
            prev_out = block(block_in)
        hx = prev_out
        
        if self.image_fusion:
            attention_map = torch.sigmoid(3.0 * self.conv_attention(hx))
            output = attention_map * hr_comp + (1.0 - attention_map) * self.to_rgb(hx)
        else:
            output = self.to_rgb(hx)

        return output

class DBPNet_official(nn.Module):
    def __init__(self, num_channels=3, base_filter=64, feat=256, image_fusion=True):
        super(DBPNet_official, self).__init__()
        
        kernel = 6
        stride = 2
        padding = 2
        self.image_fusion = image_fusion

        #Initial Feature Extraction
        self.in_conv = ConvBlock(7, feat, 3, 1, 1, activation=nn.PReLU, norm_layer=None)
        self.fea_conv = ConvBlock(32, feat, 3, 1, 1, activation=nn.PReLU, norm_layer=None)
        self.feat1 = ConvBlock(feat, base_filter, 1, 1, 0, activation=nn.PReLU, norm_layer=None)
        #Back-projection stages
        self.up1 = UpBlock(base_filter, kernel, stride, padding)
        self.down1 = DownBlock(base_filter, kernel, stride, padding)
        #Reconstruction
        if self.image_fusion:
            self.conv_attention = nn.Conv2d(base_filter, 1, kernel_size=1)
        self.to_rgb = nn.Conv2d(base_filter, 3, 1,1,0)

        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif classname.find('ConvTranspose2d') != -1:
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
        
            
    def forward(self, lr_output, hr_mask, hr_comp, extra_feat=None, lut_hr_output=None, target_resolution=(1024,1024)):
        upsampled_output = F.interpolate(lr_output, size=target_resolution, mode='bilinear')
        extra_feat = F.interpolate(extra_feat, size=target_resolution, mode='bilinear') # C=32
        images = torch.cat((upsampled_output, lut_hr_output, hr_mask),1)
        x = self.in_conv(images)+self.fea_conv(extra_feat)
        x = self.feat1(x)
        
        h1 = self.up1(x)
        d1 = self.down1(h1)
        
        if self.image_fusion:
            attention_map = torch.sigmoid(3.0 * self.conv_attention(d1))
            output = attention_map * hr_comp + (1.0 - attention_map) * self.to_rgb(d1)
        else:
            output = self.to_rgb(d1)
        
        return output


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class spatial_att_refine(nn.Module):
    def __init__(self):
        super(spatial_att_refine, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, lr_output, hr_mask, hr_comp, extra_feat=None, lut_hr_output=None, target_resolution=(1024,1024)):
        upsampled_output = F.interpolate(lr_output, size=target_resolution, mode='bilinear')
        rrt_images = torch.cat((upsampled_output, hr_mask),1)
        ppt_images = torch.cat((lut_hr_output, hr_mask),1)
        rrt_compress = self.compress(rrt_images)
        rrt_out = self.spatial(rrt_compress)
        rrt_scale = F.sigmoid(rrt_out) # broadcasting
        ppt_compress = self.compress(ppt_images)
        ppt_out = self.spatial(ppt_compress)
        ppt_scale = F.sigmoid(ppt_out)
        output = upsampled_output*rrt_scale+lut_hr_output*ppt_scale
        return output
