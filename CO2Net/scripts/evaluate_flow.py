import os
import cv2
import numpy as np
import torch
import sys
sys.path.insert(0, '.')
from flownet import *
from flownet.resample2d_package.resample2d import Resample2d
import os
import time
import argparse
from skimage.measure import compare_mse as mse
from iharm.data.transforms import HCompose, LongestMaxSizeIfLarger
from albumentations import Resize, NoOp
import argparse
crop_size = (256, 256)
val_augmentator = HCompose([Resize(*crop_size)])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default=None,
                        help='')
    parser.add_argument('--dataset_path_next', type=str, default=None,
                        help='')
    parser.add_argument('--cur_result', type=str, default=None,
                        help='')
    parser.add_argument('--next_result', type=str, default=None,
                        help='')
    args = parser.parse_args()
    return args


args = parse_args()

args.rgb_max = 255.0
args.fp16 = False
net = FlowNet2(args, requires_grad=False)
checkpoint = torch.load("./flownet/FlowNet2_checkpoint.pth.tar")
net.load_state_dict(checkpoint['state_dict'])
net=net.cuda()

flow_warp = Resample2d()
flow_warp=flow_warp.cuda()
tasks = []
#cur_dir = '/new_data/result_rain_8_8'
next_tar_dir = args.dataset_path_next
#cur_dir = '/new_data/result_issam'
cur_dir = args.cur_result
next_dir = args.next_result
cur_tar_dir = args.dataset_path

mean = (0.485 * 255, 0.456 * 255, 0.406 * 255)
std = (0.229 * 255, 0.224 * 255, 0.225 * 255)

mean = torch.tensor([.485*255, .456*255, .406*255], dtype=torch.float32).view(1, 3, 1, 1).cuda(
)
std = torch.tensor([.229*255, .224*255, .225*255], dtype=torch.float32).view(1, 3, 1, 1).cuda()

final_tasks = set([])
f = open('tl_task.txt', 'r')
for line in f.readlines():
    line = line.strip()
    final_tasks.add(line)


def save_image2(bgr_image, result_name):
    torch.clamp(bgr_image, 0, 1)
    bgr_image = bgr_image.detach().cpu().numpy() * 255
    bgr_image = bgr_image.astype(np.uint8)
    bgr_image = np.transpose(bgr_image, (1, 2, 0))
    if bgr_image.shape[0] == 1:
        cv2.imwrite(
            result_name,
            bgr_image,
            [cv2.IMWRITE_JPEG_QUALITY, 85]
        )
        return
    #rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_RGB2BGR)
    rgb_image = bgr_image
    cv2.imwrite(
        result_name,
        rgb_image,
        [cv2.IMWRITE_JPEG_QUALITY, 85]
    )


def save_image(bgr_image, result_name):
    torch.clamp(bgr_image, 0, 1)
    bgr_image = bgr_image.detach().cpu().numpy() * 255
    bgr_image = bgr_image.astype(np.uint8)
    bgr_image = np.transpose(bgr_image, (1, 2, 0))
    if bgr_image.shape[0] == 1:
        cv2.imwrite(
            result_name,
            bgr_image,
            [cv2.IMWRITE_JPEG_QUALITY, 85]
        )
        return
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_RGB2BGR)
    rgb_image = rgb_image
    cv2.imwrite(
        result_name,
        rgb_image,
        [cv2.IMWRITE_JPEG_QUALITY, 85]
    )


with open('./test_frames.txt', 'r') as f:
    for line in f.readlines():
        line = line.strip()
        tasks.append(line)
total_fmse = 0
total_ori_fmse = 0
t1 = time.time()
or_tls = []
max_ori = 0
count = 0
for i, task in enumerate(tasks):
    task = task.strip()
    if task not in final_tasks:
        continue
    count += 1
    if i % 100 == 0:
        t2 =time.time()

        print(i, t2-t1)
        t1 = time.time()

    video, obj, img_num = task.split()[1].split('/')[-3:]
    cur_name = video + '_' + obj + '_' + img_num[:-3] + 'npy'
    cur_name_without_obj = video + '_' + img_num[:-3] + 'npy'
    next_name = video + '_' + obj + '_' + '%05d' % (int(img_num[:-4]) + 1) + '.npy'
    next_name_without_obj = video + '_' + '%05d' % (int(img_num[:-4]) + 1) + '.npy'
    cur_target_name = os.path.join(cur_tar_dir, task.split()[0])

    cur_original_pic = torch.from_numpy(val_augmentator(image=cv2.imread(cur_target_name))["image"][:, :, ::-1].transpose(2, 0, 1).copy()).cuda().unsqueeze(0).float()
    used_cur_original_pic = cur_original_pic / 255

    pre, obj, num = task.split()[0].split('/')
    num = '%05d' % (int(num[:-4]) + 1) + num[-4:]
    next_tar_name = os.path.join(next_tar_dir, pre + '/' + obj + '/' + num)
    assert os.path.exists(next_tar_name)

    next_original_pic = torch.from_numpy(val_augmentator(image=cv2.imread(next_tar_name))["image"][:, :, ::-1].transpose(2, 0, 1).copy()).cuda().unsqueeze(0).float()

    used_next_original_pic = next_original_pic / 255
    cur_tensor_name = os.path.join(cur_dir, cur_name)
    cur_tensor_name = cur_tensor_name[:-4] + '.npy'
    cur_pic = torch.from_numpy(np.load(cur_tensor_name)).cuda().float()

    next_tensor_name = os.path.join(next_dir, next_name)
    next_tensor_name = next_tensor_name[:-4] + '.npy'

    next_pic = torch.from_numpy(np.load(next_tensor_name)).cuda().float()

    cur_mask = cv2.cvtColor(cv2.imread(os.path.join(cur_tar_dir, task.split()[1])), cv2.COLOR_BGR2RGB)[:, :, 0].astype(np.float32) / 255.
    cur_mask = val_augmentator(object_mask=cur_mask, image=cv2.imread(next_tar_name))['object_mask']
    cur_mask = torch.from_numpy(cur_mask).cuda().unsqueeze(0)


    with torch.no_grad():
        flow = net(next_original_pic, cur_original_pic)

    cur_mask = torch.reshape(cur_mask, (1, 1, 256, 256))
    warp_cur_tensor = flow_warp(cur_pic, flow)
    ori_warp_cur_tensor = flow_warp(used_cur_original_pic, flow)

    warp_cur_mask = flow_warp(cur_mask, flow)
    dif = torch.exp(-torch.abs(ori_warp_cur_tensor - used_next_original_pic))
    dif = torch.sum(dif, dim = 1)/3
    dif =torch.reshape(dif, (1,1,256,256))
    final_mask = warp_cur_mask *dif
    fmse = ((warp_cur_tensor * final_mask - next_pic*final_mask)**2).sum() * 255 * 255 / final_mask.sum()
    fmse_ori = ((ori_warp_cur_tensor * final_mask - used_next_original_pic*final_mask)**2).sum() * 255 * 255 / final_mask.sum()
    total_fmse += fmse
    total_ori_fmse += fmse_ori

print("in total {} pairs, current tl loss is {} and original tl loss is {}".format(len(final_tasks),
                                                                                   "%.2f" % (float(total_fmse.detach().cpu().numpy()) / len(final_tasks)),
                                                                                   "%.2f" % (float(total_ori_fmse.detach().cpu().numpy()) / len(final_tasks))))
