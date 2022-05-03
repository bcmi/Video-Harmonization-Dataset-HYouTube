import argparse
import sys

sys.path.insert(0, '.')

import torch
import os
import time
import numpy as np
import cv2
from iharm.utils import pytorch_ssim


from iharm.utils.misc import load_weights
from skimage.measure import compare_mse as mse
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ssim
from tqdm import tqdm
from torch.utils.data import DataLoader
from iharm.data.compose import ComposeDataset, MyPreviousDataset, MyPreviousSequenceDataset
from pathlib import Path
from iharm.inference.transforms import NormalizeTensor, PadToDivisor, ToTensor, AddFlippedTensor
from torchvision import transforms
from iharm.model.base import SSAMvideoLutWithoutDbp, SSAMvideoLut
from albumentations import Resize, NoOp
from iharm.data.hdataset import HDataset
from iharm.data.transforms import HCompose, LongestMaxSizeIfLarger
from iharm.inference.predictor import Predictor
from iharm.inference.evaluation import evaluate_dataset
from iharm.inference.metrics import MetricsHub, MSE, fMSE, PSNR, N, AvgPredictTime
from iharm.inference.utils import load_model, find_checkpoint
from iharm.mconfigs import ALL_MCONFIGS
from iharm.utils.exp import load_config_file
from iharm.utils.log import logger, add_new_file_output_to_logger








tmp_transform = [
        ToTensor(torch.device(int(4))),
        #transforms.Normalize(model_cfg.input_normalization['mean'], model_cfg.input_normalization['std']),
    ]


def _save_image(bgr_image, result_name):
    #print(bgr_image.max(), bgr_image.min(), bgr_image.shape)
    bgr_image = torch.clamp(bgr_image, 0, 1)
    bgr_image = bgr_image.detach().cpu().numpy()* 255
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
    cv2.imwrite(
        result_name,
        rgb_image,
        [cv2.IMWRITE_JPEG_QUALITY, 85]
    )



RESIZE_STRATEGIES = {
    'None': NoOp(),
    'LimitLongest1024': LongestMaxSizeIfLarger(1024),
    'Fixed256': Resize(256, 256),
    'Fixed512': Resize(512, 512)
}


def parse_args():
    parser = argparse.ArgumentParser()
    '''
    parser.add_argument('checkpoint', type=str,
                        help='The path to the checkpoint. '
                             'This can be a relative path (relative to cfg.MODELS_PATH) '
                             'or an absolute path. The file extension can be omitted.')
    '''
    parser.add_argument('--resize-strategy', type=str, choices=RESIZE_STRATEGIES.keys(), default='Fixed256')
    parser.add_argument('--use-flip', action='store_true', default=False,
                        help='Use horizontal flip test-time augmentation.')
    parser.add_argument('--use_feature', action='store_true', default=False,
                        help='Use horizontal flip test-time augmentation.')

    parser.add_argument('--backbone_type', type=str, default='issam',
                        help='Use horizontal flip test-time augmentation.')

    parser.add_argument('--write_lut_output', type=str, default="",
                        help='directory of write lut results for training')
    parser.add_argument('--write_lut_map', type=str, default="",
                        help='directory of write lut results for training')

    parser.add_argument('--normalize_outside', action='store_true', default=False,
                        help='Use horizontal flip test-time augmentation.')

    parser.add_argument('--gpu', type=str, default=0, help='ID of used GPU.')
    parser.add_argument('--config-path', type=str, default='./config.yml',
                        help='The path to the config file.')

    parser.add_argument('--eval-prefix', type=str, default='')
    parser.add_argument('--train_list', type=str, default=None,
                        help='Model weights will be loaded from the specified path if you use this argument.')

    parser.add_argument('--val_list', type=str, default=None,
                        help='Model weights will be loaded from the specified path if you use this argument.')

    parser.add_argument('--write_npy_backbone', action='store_true',
                        help='write npy backbone reult')
    parser.add_argument('--write_npy_result', action='store_true',
                        help='write npy final result.')
    parser.add_argument('--backbone_npy_dir', type=str, default="")
    parser.add_argument('--result_npy_dir', type=str, default="")


    parser.add_argument('--dataset_path', type=str, default=None,
                        help='')
    parser.add_argument('--previous_num', type=int, default=1)
    parser.add_argument('--future_num', type=int, default=1)
    parser.add_argument('--backbone', type=str, default="", help='')
    parser.add_argument("--checkpoint", type=str, default="", help='')
    args = parser.parse_args()
    return args

def denormalize(tensor):
    mean = torch.tensor([.485, .456, .406], dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor([.229, .224, .225], dtype=torch.float32).view(1, 3, 1, 1)
    mean = mean.to(tensor.device)
    std = std.to(tensor.device)
    #self.init_device(tensor.device)
    return tensor * std + mean


def main(given_number):
    t1 = time.time()
    args = parse_args()
    #checkpoint_path = find_checkpoint(cfg.MODELS_PATH, args.checkpoint)
    previous_number = args.previous_num
    future_number = args.future_num
    #previous_number = given_number
    #future_number = given_number
    crop_size = (256, 256)
    val_augmentator = HCompose([
        Resize(*crop_size)
    ])
    logger.info(vars(args))

    device = torch.device(f'cuda:{args.gpu}')
    #net = load_model(args.model_type, checkpoint_path, verbose=True)

    if args.checkpoint:
        net = SSAMvideoLut(
            depth=4, backbone_path=args.backbone, with_lutoutput=False, need_normalize=not args.normalize_outside,
            need_denormalize=not args.normalize_outside, use_feature=args.use_feature, backbone_type = args.backbone_type
        )
        load_weights(net, args.checkpoint)
        print("load checkpoint")
    else:
        net = SSAMvideoLutWithoutDbp(
            depth=4, backbone_path = args.backbone, fix_threshold = 0.1, k_threshold = 0.1, use_lutoutput=False,
            write_dir = args.write_dir ,backbone_type = args.backbone_type)

    net.eval()
    net.to(torch.device(int(args.gpu)))
    if args.normalize_outside:
        mean = [.485, .456, .406]
        std = [.229, .224, .225]
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        input_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    valset = MyPreviousSequenceDataset(
        args.val_list, args.dataset_path, previous_number, future_number,
        augmentator=val_augmentator,
        input_transform=input_transform,
        keep_background_prob=-1, with_previous=True
    )

    

    batch_size = 32
    valdata = DataLoader(
            valset,  batch_size, shuffle=False,
            drop_last=False, pin_memory=False,
            num_workers=8
        )

    original_mse = 0
    original_fmse = 0
    cur_to_tar = 0
    cur_to_iss = 0
    cur_to_iss_f = 0
    backbone_to_tar = 0
    cur_psnr = 0
    total_fore_ground = 0
    backbone_psnr = 0
    fmse_tar = 0
    fmse = 0
    total = 0
    cur_ssim = 0
    backbone_ssim = 0
    total_me = 0
    tbar = tqdm(valdata, ncols=80)

    original_psnr = 0
    invalid_ratio = 0
    start_time = time.time()

    for q, batch_data in enumerate(tbar):
        names = batch_data['name']

        #continue
        batch_data = {k: v.to(torch.device(int(args.gpu))) for k, v in batch_data.items() if k != 'name'}
        images, masks = batch_data['images'], batch_data['masks']
        target_images = batch_data['target_images']
        total += target_images.shape[0]
        previous = {'images': batch_data['pre_images'], 'masks': batch_data['pre_masks']}


        direct_lutoutput = batch_data['lut_output'] if 'lut_output' in batch_data else None
        outs = net(images, masks, previous=previous, names=names, direct_lutoutput=direct_lutoutput, direct_lut_map = None)

        if args.normalize_outside:
            outs['images'] = denormalize(outs['images'])
            target_images = denormalize(target_images)

        predicted_images = outs['images']
        backbone_outs = outs["backbone_out"]

        
        for b in range(images.shape[0]):
            video, obj, img_number = names[b].split('/')[-3:]
            write_name = video + '_' + obj + '_' + img_number
            '''
            _save_image(predicted_images[b], './try_result/ours/' + write_name)
            _save_image(backbone_outs[b], './try_result/backbone/' + write_name)
            _save_image(images[b], './try_result/ori/' + write_name)
            _save_image(target_images[b], './try_result/gt/' + write_name)
            '''
            _save_image(backbone_outs[b], './try_result/huang/' + write_name)

        if not args.checkpoint:
            total_me += outs["me"]
            invalid_ratio += outs["invalid"]
        for b in range(images.shape[0]):
            ssim_score, fssim_score = pytorch_ssim.ssim(torch.clamp(predicted_images[b:b + 1, :, :, :] * 255, 0, 255),
                                                        torch.clamp(target_images[b:b + 1, :, :, :] * 255, 0, 255),
                                                        window_size=11, mask=masks[b:b + 1, :, :, :])
            cur_ssim += fssim_score
            backbone_ssim_score, backbone_fssim_score = pytorch_ssim.ssim(
                torch.clamp(backbone_outs[b:b + 1, :, :, :] * 255, 0, 255),
                torch.clamp(target_images[b:b + 1, :, :, :] * 255, 0, 255), window_size=11,
                mask=masks[b:b + 1, :, :, :])
            backbone_ssim += backbone_fssim_score


        if args.write_npy_backbone:
            for b in range(images.shape[0]):
                video, obj, img_number = names[b].split('/')[-3:]
                backbone_name = os.path.join(
                    args.backbone_npy_dir,
                    video + '_' + obj + '_' + img_number[:-3] + 'npy')
                np.save(backbone_name, backbone_outs[b:b + 1, :, :, :].detach().cpu().numpy())
        if args.write_npy_result:
            for b in range(images.shape[0]):
                video, obj, img_number = names[b].split('/')[-3:]
                result_name = os.path.join(
                    args.result_npy_dir,
                    video + '_' + obj + '_' + img_number[:-3] + 'npy')
                np.save(result_name, predicted_images[b:b + 1, :, :, :].detach().cpu().numpy())

        for i in range(predicted_images.shape[0]):
            mask = masks[i]
            for transform in reversed(tmp_transform):
                with torch.no_grad():
                    image = transform.inv_transform(images[i])
                    predicted_image = transform.inv_transform(predicted_images[i])
                    target_image = transform.inv_transform(target_images[i])
                    backbone_out = transform.inv_transform(backbone_outs[i])

            image = torch.clamp(image, 0, 255)
            image = image.cpu().numpy()



            predicted_image = torch.clamp(predicted_image, 0, 255)
            predicted_image = predicted_image.cpu().numpy()



            target_image = torch.clamp(target_image, 0, 255)
            target_image = target_image.cpu().numpy()

            mask = mask.cpu().numpy()
            mask = mask.astype(np.uint8)
            fore_ground = mask.sum()
            total_fore_ground += fore_ground

            backbone_out = torch.clamp(backbone_out, 0, 255)
            backbone_out = backbone_out.cpu().numpy()



            cur_psnr += psnr(predicted_image, target_image, data_range = predicted_image.max() - predicted_image.min())
            backbone_psnr += psnr(backbone_out, target_image, data_range = backbone_out.max() - backbone_out.min())
            original_psnr += psnr(image, target_image, data_range = image.max() - image.min())

            original_mse += mse(image, target_image)
            original_fmse += mse(image, target_image) * 256 * 256 / fore_ground
            fmse += mse(backbone_out, target_image) * 256 * 256 / fore_ground
            fmse_tar += mse(predicted_image, target_image) * 256 * 256 / fore_ground
            cur_to_tar += mse(predicted_image, target_image)
            cur_to_iss += mse(predicted_image, backbone_out)
            cur_to_iss_f += mse(predicted_image, backbone_out) * 256 * 256 / fore_ground
            backbone_to_tar += mse(backbone_out, target_image)


    print(previous_number, future_number)
    print("backbone  fmse", fmse / total)
    print("backbone mse:", backbone_to_tar/total)
    print("current fmse:", fmse_tar / total)
    print("current mse:", cur_to_tar / total)
    print("backbone psnr:", backbone_psnr / total, "backbone ssim:", backbone_ssim / total)
    print("current psnr:", cur_psnr / total, "current ssim:", cur_ssim / total)
    print("me:", total_me / total)
    print("invalid ratio:", invalid_ratio / total)






    end_time = time.time()
    print("cost:", end_time - start_time)

if __name__ == '__main__':
    numbers = [1]
    for number in numbers:
        main(number)
