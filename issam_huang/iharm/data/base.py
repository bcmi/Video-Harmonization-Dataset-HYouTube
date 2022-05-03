import random
import numpy as np
import torch


class BaseHDataset(torch.utils.data.dataset.Dataset):
    def __init__(self,
                 augmentator=None,
                 input_transform=None,
                 keep_background_prob=0.0,
                 with_image_info=False,
                 epoch_len=-1):
        super(BaseHDataset, self).__init__()
        self.epoch_len = epoch_len
        self.input_transform = input_transform
        self.augmentator = augmentator
        self.keep_background_prob = keep_background_prob
        self.with_image_info = with_image_info

        if input_transform is None:
            input_transform = lambda x: x
        print(input_transform)
        self.input_transform = input_transform
        self.dataset_samples = None

    def __getitem__(self, index):
        if self.epoch_len > 0:
            index = random.randrange(0, len(self.dataset_samples))

        # print(111111111111111111111111)
        sample = self.get_sample(index)
        # print(222222222222222222222222222222)
        self.check_sample_types(sample)
        # print(33333333333333333333333333333)
        sample = self.augment_sample(sample)
        # print(444444444444444444444444)
        # print(2222222222222222222222222222222)

        # print(sample['image_pre'].shape,sample['object_mask_pre'].shape,sample['image'].shape,sample['target_image'].shape,sample['object_mask'].shape)
        image = self.input_transform(sample['image'])
        target_image = self.input_transform(sample['target_image'])
        obj_mask = sample['object_mask'].astype(np.float32)

        # print(33333333333333333333333333333333)

        image_pre = self.input_transform(sample['image_pre'])
        obj_mask_pre = sample['object_mask_pre'].astype(np.float32)

        # print(44444444444444444444444444444)

        # print(image_pre.shape,image.shape)
        output = {
            'images_pre': image_pre,
            'masks_pre': obj_mask_pre[np.newaxis, ...].astype(np.float32),
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
        # print(self.augmentator.additional_targets.keys())
        additional_targets = {target_name: sample[target_name]
                              for target_name in self.augmentator.additional_targets.keys()}
        # print(additional_targets.keys())
        valid_augmentation = False
        while not valid_augmentation:
            aug_output = self.augmentator(image=sample['image'], **additional_targets)
            valid_augmentation = self.check_augmented_sample(sample, aug_output)

        for target_name, transformed_target in aug_output.items():
            # print(target_name)
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
