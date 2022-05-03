import argparse
import sys

sys.path.insert(0, '.')

import torch
import os
from functools import partial
import time
import numpy as np
import cv2
from iharm.utils.misc import load_weights
from skimage.measure import compare_mse as mse
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ssim
from tqdm import tqdm
from torch.utils.data import DataLoader
from iharm.data.compose import ComposeDataset, MyPreviousDataset, MyPreviousSequenceDataset, MyDirectDataset
from pathlib import Path
from iharm.inference.transforms import NormalizeTensor, PadToDivisor, ToTensor, AddFlippedTensor
from torchvision import transforms
from iharm.model.base import SSAMvideoLutWithoutDbp, SSAMvideoLut, SSAMImageHarmonization
from albumentations import Resize, NoOp
from iharm.model import initializer
from iharm.data.transforms import HCompose, LongestMaxSizeIfLarger
from iharm.utils import misc
import math
from iharm.data.hdataset import HDataset
from iharm.data.transforms import HCompose, LongestMaxSizeIfLarger
from iharm.inference.predictor import Predictor
from iharm.inference.evaluation import evaluate_dataset
from iharm.inference.metrics import MetricsHub, MSE, fMSE, PSNR, N, AvgPredictTime
from iharm.inference.utils import load_model, find_checkpoint
from iharm.mconfigs import ALL_MCONFIGS
from iharm.utils.exp import load_config_file
from iharm.utils.log import logger, add_new_file_output_to_logger

#f = open("./refine_without_feature_withoutnoramlized/logs.txt", 'a')



def MaskWeight_MSE(pred, label, mask):
    label = label.view(pred.size())
    reduce_dims = misc.get_dims_with_exclusion(label.dim(), 0)
    loss = (pred - label) ** 2
    delimeter = pred.size(1) * torch.sum(mask, dim=reduce_dims)
    loss = torch.sum(loss, dim=reduce_dims) / delimeter
    loss = torch.mean(loss)

    return loss


def parse_args():
    parser = argparse.ArgumentParser()
    '''
    parser.add_argument('checkpoint', type=str,
                        help='The path to the checkpoint. '
                             'This can be a relative path (relative to cfg.MODELS_PATH) '
                             'or an absolute path. The file extension can be omitted.')
    '''
    #parser.add_argument('--resize-strategy', type=str, choices=RESIZE_STRATEGIES.keys(), default='Fixed256')
    parser.add_argument('--gpu', type=str, default=0, help='ID of used GPU.')
    parser.add_argument('--config-path', type=str, default='./config.yml',
                        help='The path to the config file.')
    parser.add_argument('--backbone_type', type=str, default='issam',
                        help='Use horizontal flip test-time augmentation.')
    parser.add_argument('--eval-prefix', type=str, default='')
    parser.add_argument('--train_list', type=str, default=None,
                        help='Model weights will be loaded from the specified path if you use this argument.')

    parser.add_argument('--val_list', type=str, default=None,
                        help='Model weights will be loaded from the specified path if you use this argument.')

    parser.add_argument('--dataset_path', type=str, default=None,
                        help='')

    parser.add_argument('--exp_name', type=str, default=None,
                        help='')
    parser.add_argument('--previous_num', type=int, default=1)
    parser.add_argument('--future_num', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--backbone', type=str, default="", help='')
    parser.add_argument('--use_feature', action='store_true', default=False, help='')
    parser.add_argument('--normalize_inside', action='store_true', default=False, help='')
    parser.add_argument('--lut_map_dir', default="", help='direct result of lut map')
    parser.add_argument('--lut_output_dir', default="", help='direct result of lut output')
    parser.add_argument("--checkpoint", type=str, default="", help='')

    args = parser.parse_args()
    return args


def evaluate(net, val_dataloader, args, epoch):
    net.eval()
    tbar = tqdm(val_dataloader, ncols=70)
    total_loss = 0.0
    for q, batch_data in enumerate(tbar):

        names = batch_data['name']
        batch_data = {k: v.to(torch.device(int(args.gpu))) for k, v in batch_data.items() if k != 'name'}
        images, masks = batch_data['images'], batch_data['masks']
        target_images = batch_data['target_images']
        if len(args.lut_output_dir) > 0:
            outs = net(images, masks, previous=None, names=names, direct_lutoutput=batch_data['lut_output'],
                       direct_lut_map=batch_data['lut_map'])
        else:
            previous = {'images': batch_data['pre_images'], 'masks': batch_data['pre_masks']}
            outs = net(images, masks, previous=previous, names=names, direct_lutoutput=None, direct_lut_map=None)
        loss = MaskWeight_MSE(outs['images'], target_images, masks)
        total_loss += loss.item()
        if q % 100 == 0:
            print("------------")
            with open("./{}/logs.txt".format(args.exp_name), 'a') as f:
                f.write("val\t"+"epoch {}:\t".format(epoch) + str(q + 1) + '\t' + str(total_loss / (q + 1)) + '\n')
            print("val\t"+"epoch {}:\t".format(epoch) + str(q + 1) + '\t' + str(total_loss / (q + 1)) + '\n')
    print("final loss" , "epoch {}:\t".format(epoch), total_loss / (q+1) * 255 * 255)
    with open("./{}/logs.txt".format(args.exp_name), 'a') as f:
        f.write("val\t" +"epoch {}:\t".format(epoch) + "final loss\t" + str(total_loss / (q+1) * 255 * 255) + '\n')


def main():
    args = parse_args()
    net = SSAMvideoLut(
        depth=4, backbone_path=args.backbone, with_lutoutput=True, need_normalize = args.normalize_inside, need_denormalize =  args.normalize_inside,
        use_feature=args.use_feature, backbone_type = args.backbone_type
    )
    net.to(torch.device(int(args.gpu)))
    crop_size = (256, 256)
    augmentator = HCompose([
        Resize(*crop_size)
    ])

    params = []
    for name, param in net.named_parameters():
        param_group = {'params': [param]}
        if not param.requires_grad:
            print("not grad")
            params.append(param_group)
            continue
        if not math.isclose(getattr(param, 'lr_mult', 1.0), 1.0):
            #logger.info(f'Applied lr_mult={param.lr_mult} to "{name}" parameter.')
            param_group['lr'] = param_group.get('lr', base_lr) * param.lr_mult
        params.append(param_group)

    optimizer_params = {
        'lr': 1e-5,
        'betas': (0.9, 0.999), 'eps': 1e-8
    }

    alpha = 2.0
    optimizer = torch.optim.AdamW(params, **optimizer_params)
    net.load_backbone()
    if len(args.checkpoint) > 0:
        load_weights(net, args.checkpoint)
        print("load checkpoint")
    lr_scheduler = partial(torch.optim.lr_scheduler.MultiStepLR,
                           milestones=[105, 115], gamma=0.1)(optimizer=optimizer)

    if args.normalize_inside:
        input_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    else:
        print('normal outside')
        mean =  [.485, .456, .406]
        std = [.229, .224, .225]

        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])


    previous_number = args.previous_num
    future_number = args.future_num
    if len(args.lut_map_dir) > 0:
        valset = MyDirectDataset(args.val_list, args.dataset_path, backbone_type=args.backbone_type,
                                    input_transform=input_transform,
                                    augmentator=augmentator, lut_map_dir=args.lut_map_dir, lut_output_dir=args.lut_output_dir)


        trainset = MyDirectDataset(args.train_list, args.dataset_path, backbone_type=args.backbone_type,
                                  input_transform=input_transform,
                                  augmentator=augmentator, lut_map_dir=args.lut_map_dir, lut_output_dir=args.lut_output_dir)
    else:
        valset = MyPreviousSequenceDataset(
            args.val_list, args.dataset_path, previous_number, future_number,
            augmentator=val_augmentator,
            input_transform=input_transform,
            keep_background_prob=-1, with_previous=True
        )
        trainset = MyPreviousSequenceDataset(
            args.trainset, args.dataset_path, previous_number, future_number,
            augmentator=val_augmentator,
            input_transform=input_transform,
            keep_background_prob=-1, with_previous=True
        )


    batch_size = 32

    val_dataloader = DataLoader(
            valset,  batch_size, shuffle=False,
            drop_last=False, pin_memory=True,
            num_workers=8
         )

    train_dataloader = DataLoader(
        trainset, batch_size, shuffle=True,
        drop_last=False, pin_memory=True,
        num_workers=8
    )
    if not os.path.exists(args.exp_name):
        os.mkdir(args.exp_name)
    for epoch in range(args.epochs):
        net.train()
        tbar = tqdm(train_dataloader, ncols=70)
        total_loss = 0.0
        for q, batch_data in enumerate(tbar):
            names = batch_data['name']
            batch_data = {k: v.to(torch.device(int(args.gpu))) for k, v in batch_data.items() if k != 'name'}
            images, masks = batch_data['images'], batch_data['masks']
            target_images = batch_data['target_images']
            if len(args.lut_output_dir) > 0:
                outs = net(images, masks, previous=None, names=names, direct_lutoutput=batch_data['lut_output'], direct_lut_map = batch_data['lut_map'])
            else:
                previous = {'images': batch_data['pre_images'], 'masks': batch_data['pre_masks']}
                outs = net(images, masks, previous=previous, names=names, direct_lutoutput=None, direct_lut_map=None)
            loss = MaskWeight_MSE(outs['images'], target_images, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            lr_scheduler.step()
            if q%100 == 0:
                print("------------")
                with open("./{}/logs.txt".format(args.exp_name), 'a') as f:
                    f.write("train\t" + "epoch {}:\t".format(epoch) +str(q+1) + '\t' + str(total_loss / (q+1)) + '\n')
                print("train\t" +"epoch {}:\t".format(epoch) + str(q+1) + '\t' + str(total_loss / (q+1)) + '\n')
        print("epoch {}:".format(epoch), total_loss)
        torch.save(net.state_dict(), './{}/{}.pth'.format(args.exp_name, epoch+1))
        torch.save(net.state_dict(), './{}/last_checkpoint.pth'.format(args.exp_name))
        evaluate(net, val_dataloader, args, epoch)

main()