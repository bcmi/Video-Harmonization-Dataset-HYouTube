import argparse
import sys

sys.path.insert(0, '.')

import torch
from tqdm import tqdm
from iharm.utils.misc import load_weights
from pathlib import Path
from skimage.measure import compare_mse as mse
from albumentations import Resize, NoOp
from iharm.inference.transforms import NormalizeTensor, PadToDivisor, ToTensor, AddFlippedTensor
from iharm.data.hdataset import HDataset, MyDataset
from iharm.data.transforms import HCompose, LongestMaxSizeIfLarger
from iharm.inference.predictor import Predictor
from iharm.inference.evaluation import evaluate_dataset
from iharm.inference.metrics import MetricsHub, MSE, fMSE, PSNR, N, AvgPredictTime
from iharm.inference.utils import load_model, find_checkpoint
from iharm.mconfigs import ALL_MCONFIGS
from torch.utils.data import DataLoader
from iharm.model.base import SSAMImageHarmonization
from iharm.utils.exp import load_config_file
from iharm.utils.log import logger, add_new_file_output_to_logger

RESIZE_STRATEGIES = {
    'None': NoOp(),
    'LimitLongest1024': LongestMaxSizeIfLarger(1024),
    'Fixed256': Resize(256, 256),
    'Fixed512': Resize(512, 512)
}


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--resize-strategy', type=str, choices=RESIZE_STRATEGIES.keys(), default='Fixed256')
    parser.add_argument('--use-flip', action='store_true', default=False,
                        help='Use horizontal flip test-time augmentation.')
    parser.add_argument('--gpu', type=str, default=0, help='ID of used GPU.')
    parser.add_argument('--config-path', type=str, default='./config.yml',
                        help='The path to the config file.')




    parser.add_argument('--eval-prefix', type=str, default='')

    parser.add_argument('--val_list', type=str, default=None,
                        help='Model weights will be loaded from the specified path if you use this argument.')

    parser.add_argument('--dataset_path', type=str, default=None,
                        help='')

    parser.add_argument('--ssam_backbone', type=str, default="", help='')
    parser.add_argument("--checkpoint", type=str, default="", help='')

    args = parser.parse_args()
    cfg = load_config_file(args.config_path, return_edict=True)
    return args, cfg


def main():
    args, cfg = parse_args()
    checkpoint_path = find_checkpoint(cfg.MODELS_PATH, args.checkpoint)
    add_new_file_output_to_logger(
        logs_path=Path(cfg.EXPS_PATH) / 'evaluation_logs',
        prefix=f'{Path(checkpoint_path).stem}_',
        only_message=True
    )
    logger.info(vars(args))

    device = torch.device(f'cuda:{args.gpu}')
    mean = (.485, .456, .406)
    std = (.229, .224, .225)
    net = SSAMImageHarmonization(
        depth=4, ch=32, image_fusion=True, attention_mid_k=0.5,
        attend_from=2, batchnorm_from=2
    )
    net.to(device)
    size_divisor = 2 ** (net.depth + 1)
    load_weights(net, args.checkpoint)
    crop_size = (256, 256)
    mean = torch.tensor(mean, dtype=torch.float32)
    std = torch.tensor(std, dtype=torch.float32)
    input_transforms = [
        PadToDivisor(divisor=size_divisor, border_mode=0),
        ToTensor(device),
        NormalizeTensor(mean, std, device),
    ]
    val_augmentator = HCompose([
        Resize(*crop_size)
    ])
    valset = MyDataset(args.val_list, args.dataset_path,
                       augmentator=HCompose([RESIZE_STRATEGIES[args.resize_strategy]]),
                       keep_background_prob=-1)

    valdata = DataLoader(
        valset, 16, shuffle=False,
        drop_last=False, pin_memory=True,
        num_workers=4
    )

    total = 0
    total_mse = 0
    tbar = tqdm(valdata, ncols=100)
    for q, batch_data in enumerate(tbar):
        #batch_data = {k: v.to(torch.device(3)) for k, v in batch_data.items()}

        images, masks = batch_data['images'].cpu().numpy(), batch_data['masks'].cpu().numpy()
        new_images = []
        new_masks = []
        #print(masks.shape)
        total += images.shape[0]
        for i in range(images.shape[0]):
            image, mask = images[i], masks[i]
            for transform in input_transforms:
                with torch.no_grad():
                    image, mask= transform.transform(image, mask)
            new_images.append(image)
            new_masks.append(mask)
        images = torch.cat(new_images, 0).to(device)
        masks = torch.cat(new_masks, 0).squeeze(1).to(device)
        #print(images.shape, masks.shape)
        target_images = batch_data['target_images'].cpu().numpy()

        # target_images = target_images.cpu().numpy()
        # previous = {'images': batch_data['pre_images'], 'masks': batch_data['pre_masks']}
        outs = net(images, masks)
        predicted_images = outs['images']

        for i in range(predicted_images.shape[0]):
            predicted_image = predicted_images[i:i+1, :, :, :]
            #print(predicted_image.shape)
            for transform in reversed(input_transforms):
                with torch.no_grad():
                    predicted_image = transform.inv_transform(predicted_image)

            predicted_image = torch.clamp(predicted_image, 0, 255)
            predicted_image = predicted_image.cpu().numpy()
            #print(predicted_image.dtype, target_images[i].dtype)
            #print(mse(predicted_image, target_images[i]))
            total_mse += mse(predicted_image, target_images[i])
    print(total_mse/total, total_mse, total)
    '''
    datasets_names = args.datasets.split(',')
    datasets_metrics = []



    for dataset_indx, dataset_name in enumerate(datasets_names):

        dataset_metrics = MetricsHub([N(), MSE(), fMSE(), PSNR(), AvgPredictTime()],
                                     name=dataset_name)

        evaluate_dataset(dataset, predictor, dataset_metrics)
        datasets_metrics.append(dataset_metrics)
        if dataset_indx == 0:
            logger.info(dataset_metrics.get_table_header())
        logger.info(dataset_metrics)
    '''
    if len(datasets_metrics) > 1:
        overall_metrics = sum(datasets_metrics, MetricsHub([], 'Overall'))
        logger.info('-' * len(str(overall_metrics)))
        logger.info(overall_metrics)


if __name__ == '__main__':
    main()
