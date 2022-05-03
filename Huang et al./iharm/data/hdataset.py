from pathlib import Path

import cv2
import numpy as np
import os

from .base import BaseHDataset


class HDataset(BaseHDataset):
    def __init__(self, dataset_path, split, blur_target=False, **kwargs):
        super(HDataset, self).__init__(**kwargs)

        self.dataset_path = Path(dataset_path)
        self.blur_target = blur_target
        self._split = split
        self._real_images_path = self.dataset_path / 'real_images'
        self._composite_images_path = self.dataset_path / 'composite_images'
        self._masks_path = self.dataset_path / 'masks'

        images_lists_paths = [x for x in self.dataset_path.glob('*.txt') if x.stem.endswith(split)]
        assert len(images_lists_paths) == 1

        with open(images_lists_paths[0], 'r') as f:
            self.dataset_samples = [x.strip() for x in f.readlines()]

    def get_sample(self, index):
        composite_image_name = self.dataset_samples[index]
        real_image_name = composite_image_name.split('_')[0] + '.jpg'
        mask_name = '_'.join(composite_image_name.split('_')[:-1]) + '.png'

        composite_image_path = str(self._composite_images_path / composite_image_name)
        real_image_path = str(self._real_images_path / real_image_name)
        mask_path = str(self._masks_path / mask_name)

        composite_image = cv2.imread(composite_image_path)
        composite_image = cv2.cvtColor(composite_image, cv2.COLOR_BGR2RGB)

        real_image = cv2.imread(real_image_path)
        real_image = cv2.cvtColor(real_image, cv2.COLOR_BGR2RGB)

        object_mask_image = cv2.imread(mask_path)
        object_mask = object_mask_image[:, :, 0].astype(np.float32) / 255.
        if self.blur_target:
            object_mask = cv2.GaussianBlur(object_mask, (7, 7), 0)

        return {
            'image': composite_image,
            'object_mask': object_mask,
            'target_image': real_image,
            'image_id': index
        }
class MyDataset(BaseHDataset):
    def __init__(self, dataset_list, dataset_path, blur_target=False, **kwargs):
        super(MyDataset, self).__init__(**kwargs)

        self.dataset_list = dataset_list
        self.blur_target = blur_target
        self.dataset_path = dataset_path


        with open(dataset_list, 'r') as f:
            self.dataset_samples = [x.strip() for x in f.readlines()]

    def get_sample(self, index):
        real_image_name, mask_name, composite_image_name = self.dataset_samples[index].split()

        real_image_name = real_image_name.replace('\\', '/')
        mask_name = mask_name.replace('\\', '/')
        composite_image_name = composite_image_name.replace('\\', '/')

        composite_image_path = os.path.join(self.dataset_path, composite_image_name)
        real_image_path = os.path.join(self.dataset_path, real_image_name)
        mask_path = os.path.join(self.dataset_path, mask_name)

        number = composite_image_path.split('/')[-1]
        pre_number = '%05d' % (int(number[:-4]) - 5) + number[-4:]
        # print('-', number, pre_number)
        composite_image_pre_path = composite_image_path.replace(number, pre_number)
        number = mask_path.split('/')[-1]
        pre_number = '%05d' % (int(number[:-4]) - 5) + number[-4:]
        # print('=',number,pre_number)
        mask_pre_path = mask_path.replace(number, pre_number)
        if not os.path.exists(composite_image_pre_path):
            # print(composite_image_pre_path)
            composite_image_pre_path = composite_image_path
            mask_pre_path = mask_path
        #     print(11)
        # else:
        #     print(22)
        #print(composite_image_path)
        composite_image = cv2.imread(composite_image_path)
        composite_image = cv2.cvtColor(composite_image, cv2.COLOR_BGR2RGB)
        composite_image_pre = cv2.imread(composite_image_pre_path)
        composite_image_pre = cv2.cvtColor(composite_image_pre, cv2.COLOR_BGR2RGB)

        real_image = cv2.imread(real_image_path)
        real_image = cv2.cvtColor(real_image, cv2.COLOR_BGR2RGB)

        object_mask_image = cv2.imread(mask_path)
        object_mask = object_mask_image[:, :, 0].astype(np.float32) / 255.
        object_mask_image_pre = cv2.imread(mask_pre_path)
        object_mask_pre = object_mask_image_pre[:, :, 0].astype(np.float32) / 255.
        if self.blur_target:
            object_mask = cv2.GaussianBlur(object_mask, (7, 7), 0)
            object_mask_pre = cv2.GaussianBlur(object_mask_pre, (7, 7), 0)
        # print('=',composite_image_pre.shape,composite_image.shape)
        return {
            'image_pre': composite_image_pre,
            'object_mask_pre': object_mask_pre,
            'image': composite_image,
            'object_mask': object_mask,
            'target_image': real_image,
            'image_id': index
        }