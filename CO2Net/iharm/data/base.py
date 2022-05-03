import random
import numpy as np
import torch
import cv2
import os


class BaseHDataset(torch.utils.data.dataset.Dataset):
    def __init__(self,
                 augmentator=None,
                 input_transform=None,
                 keep_background_prob=0.0,
                 with_image_info=False,
                 epoch_len=-1,
                 with_previous=False):
        super(BaseHDataset, self).__init__()
        self.epoch_len = epoch_len
        self.input_transform = input_transform
        self.augmentator = augmentator
        self.keep_background_prob = keep_background_prob
        self.with_image_info = with_image_info
        self.with_previous = with_previous
        if input_transform is None:
            input_transform = lambda x: x

        self.input_transform = input_transform
        self.dataset_samples = None

    def __getitem__(self, index):
        if self.epoch_len > 0:
            index = random.randrange(0, len(self.dataset_samples))
        sample = self.get_sample(index)
        self.check_sample_types(sample)
        #print(sample['pre_image'].shape)
        pre_imgs = []
        pre_masks = []
        names = sample['name']
        if self.with_previous:
            for index in range(len(sample['pre_image'])):
                #print("--------",np.unique(sample['pre_object_mask'][index]), np.unique(sample['object_mask'][index]))
                #print(sample['pre_image'][index].shape, sample['image'].shape)
                tmp = self.augment_sample({
            'image': sample['image'],
            'object_mask': sample['object_mask'],
            'target_image': sample['target_image'],
            'image_id': sample['image_id'],
            'pre_image': sample['pre_image'][index],
            'pre_object_mask': sample['pre_object_mask'][index],
        })
                pre_imgs.append(tmp['pre_image'])
                pre_masks.append(tmp['pre_object_mask'])
                new_img = tmp['image']
                new_object_mask = tmp['object_mask']
                new_target_img = tmp['target_image']
        else:
            sample = self.augment_sample(sample)
        name = sample['name']
        video, obj, img_number = name.split('/')[-3:]
        

        new_img = self.input_transform(new_img)
        new_target_img = self.input_transform(new_target_img)
        new_object_mask = new_object_mask.astype(np.float32)
        for index in range(len(pre_imgs)):
            pre_imgs[index] = self.input_transform(pre_imgs[index])
            pre_masks[index] = pre_masks[index].astype(np.float32)
            pre_masks[index] = pre_masks[index][np.newaxis, ...].astype(np.float32)
        pre_imgs = torch.stack(pre_imgs, dim = 0)
        pre_masks = np.array(pre_masks)
        if self.with_previous:
            output = {
                'name':sample['name'],
                'images': new_img,
                'masks': new_object_mask[np.newaxis, ...].astype(np.float32),
                'target_images': new_target_img,
                'pre_images': pre_imgs,
                'pre_masks': pre_masks
            }
        else:
            output = {
                'name': sample['name'],
                'images': image,
                'masks': obj_mask[np.newaxis, ...].astype(np.float32),
                'target_images': target_image
            }
        if self.with_image_info and 'image_id' in sample:
            output['image_info'] = sample['image_id']


        return output

    def check_sample_types(self, sample):
        assert sample['image'].dtype == 'uint8'
        if 'target_image' in sample:
            assert sample['target_image'].dtype == 'uint8'

    def augment_sample(self, sample):
        if self.augmentator is None:
            return sample

        additional_targets = {target_name: sample[target_name]
                              for target_name in self.augmentator.additional_targets.keys()}

        valid_augmentation = False
        while not valid_augmentation:
            aug_output = self.augmentator(image=sample['image'], **additional_targets)
            valid_augmentation = self.check_augmented_sample(sample, aug_output)

        for target_name, transformed_target in aug_output.items():
            sample[target_name] = transformed_target

        return sample

    def check_augmented_sample(self, sample, aug_output):
        if self.keep_background_prob < 0.0 or random.random() < self.keep_background_prob:
            return True

        return aug_output['object_mask'].sum() > 1.0

    def get_sample(self, index):
        raise NotImplementedError

    def __len__(self):
        if self.epoch_len > 0:
            return self.epoch_len
        else:
            return len(self.dataset_samples)
