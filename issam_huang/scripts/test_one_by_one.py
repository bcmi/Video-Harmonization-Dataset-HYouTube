import argparse
import os
import os.path as osp
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
from skimage.measure import compare_mse as mse
from tqdm import tqdm

sys.path.insert(0, '.')
from iharm.inference.predictor import Predictor
from iharm.inference.utils import load_model, find_checkpoint
from iharm.mconfigs import ALL_MCONFIGS
from iharm.utils.log import logger
from iharm.utils.exp import load_config_file

def filter(total_list):
    new_list = []
    filter = {}
    with open('/home/luxinyuan/work/benchmarks/bad_object.txt', 'r') as f:
        for i in f.readlines():
            if "-" in i:
                continue
            if (len(i)<5):
                continue
            video, _, obj = i.split()[0].split('_')
            filter[video] = obj
    for sample in total_list:
        true_img_name, mask_name, com_name = sample
        video_name = mask_name.split('\\')[1]
        obj = mask_name.split('\\')[2][-1]
        if video_name in filter and filter[video_name] == obj:
            new_list.append(sample)
    print(new_list)
    return new_list





def main():
    def _save_image(image_name, bgr_image):
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(
            image_name,
            rgb_image
        )
    args, cfg = parse_args()
    gg = open('./{}_supp_result.txt'.format(args.val_list[-5]), 'w')
    #print('./{}_result.txt'.format(args.val_list.split('/')[-1]))
    device = torch.device(f'cuda:{args.gpu}')
    result_dir = '/home/luxinyuan/work/benchmarks/issam/image_harmonization/result/backbone/{}'.format(args.val_list[-5])
    checkpoint_path = find_checkpoint(cfg.MODELS_PATH, args.checkpoint)
    net = load_model(args.model_type, checkpoint_path, verbose=True)
    predictor = Predictor(net, device)
    video_count = {}

    samples = []
    with open(args.val_list, 'r')  as f:
        for line in f.readlines():
            samples.append(line.split())

    #samples = filter(samples)

    right = 0
    wrong = 0
    index = 0
    mse_original_total = 0
    mse_after_total = 0

    #logger.info(f'Save images to {cfg.RESULTS_PATH}')
    #samples = samples[:100]
    resize_shape = (args.resize, ) * 2
    for sample in tqdm(samples):
        #image_path = osp.join(args.images, image_name)

        true_img_name, mask_name, com_name = sample
        img_name = com_name.split('\\')[-1]
        obj_name = com_name.split('\\')[-2]
        video_name = com_name.split('\\')[-3]
        tar = '_'.join([video_name, obj_name])
        #print(video_name, obj_name, img_name)
        video_dir = os.path.join(result_dir, video_name)
        if not os.path.exists(video_dir):
            os.mkdir(video_dir)
        obj_dir = os.path.join(video_dir, obj_name)
        if not os.path.exists(obj_dir):
            os.mkdir(obj_dir)
        result_name = os.path.join(obj_dir, img_name)

        #print(true_img_name, mask_name, com_name)
        com_name = com_name.replace('\\', '/')
        mask_name = mask_name.replace('\\', '/')
        true_img_name = true_img_name.replace('\\', '/')
        com_name = os.path.join(args.dataset_path, com_name)
        #com_name = '/home/luxinyuan/work/benchmarks/supple' + com_name[13:]
        mask_path = os.path.join(args.dataset_path, mask_name)
        true_img_name = os.path.join(args.dataset_path, true_img_name)
        image = cv2.imread(com_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_size = image.shape
        if resize_shape[0] > 0:
            image = cv2.resize(image, resize_shape, cv2.INTER_LINEAR)

        true_img = cv2.imread(true_img_name)
        true_img = cv2.cvtColor(true_img, cv2.COLOR_BGR2RGB)
        if resize_shape[0] > 0:
            true_img = cv2.resize(true_img, resize_shape, cv2.INTER_LINEAR)
        #mask_path = osp.join(args.masks, '_'.join(image_name.split('_')[:-1]) + '.png')
        mask_image = cv2.imread(mask_path)
        if resize_shape[0] > 0:
            mask_image = cv2.resize(mask_image, resize_shape, cv2.INTER_LINEAR)

        mask = mask_image[:, :, 0]
        mask[mask <= 100] = 0
        mask[mask > 100] = 1
        mask_pixels = mask.sum()
        mask = mask.astype(np.float32)

        pred = predictor.predict(image, mask)
        #_save_image(result_name, pred)
        mse_original = mse(image, true_img)
        mse_after = mse(pred, true_img)
        gg.write(com_name + '\t' + str(mse_original) + '\t' + str(mse_after) + '\n')
        mse_after_total += mse_after
        mse_original_total += mse_original
        if mse_after < mse_original:
            video_count[tar] = video_count.get(tar, 0) + 1
            right += 1
        else:
            wrong += 1
        if index%100 == 0:
            print(index, right)
        #pre_diff = (true_img-pred)**2

        #true_img = true_img / 255
        #image = image / 255
        #pred = pred  / 255


        index += 1
    gg.close()
    for key in video_count:
        print(key, video_count[key])
    print('backbone')
    print(args.val_list)
    print(mse_original_total, mse_after_total)
    print(mse_original_total / len(samples), mse_after_total / len(samples))
    print(len(samples))

    print(right, wrong)



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_type', choices=ALL_MCONFIGS.keys())
    parser.add_argument('checkpoint', type=str,
                        help='The path to the checkpoint. '
                             'This can be a relative path (relative to cfg.MODELS_PATH) '
                             'or an absolute path. The file extension can be omitted.')
    parser.add_argument(
        '--resize', type=int, default=256,
        help='Resize image to a given size before feeding it into the network. If -1 the network input is not resized.'
    )
    parser.add_argument(
        '--original-size', action='store_true', default=False,
        help='Resize predicted image back to the original size.'
    )
    parser.add_argument('--gpu', type=str, default=0, help='ID of used GPU.')
    parser.add_argument('--config-path', type=str, default='./config.yml', help='The path to the config file.')

    parser.add_argument('--val_list', type=str, default='./config.yml',
                        help='The path to the config file.')

    parser.add_argument('--dataset_path', type=str, default='./config.yml',
                        help='The path to the config file.')

    args = parser.parse_args()
    cfg = load_config_file(args.config_path, return_edict=True)
    cfg.EXPS_PATH = Path(cfg.EXPS_PATH)

    logger.info(cfg)
    return args, cfg


if __name__ == '__main__':
    main()
