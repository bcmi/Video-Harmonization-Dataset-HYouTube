from .base import BaseHDataset
import cv2
import numpy as np
import copy
import os
import time
import torch


class MyDirectDataset:
    def __init__(self, val_list, dataset_path, backbone_type = 'issam', input_transform=None, augmentator=None, lut_map_dir='', lut_output_dir=''):
        start_time = time.time()
        self.tasks = []
        self.dataset_path = dataset_path
        self.input_transform = input_transform
        self.backbone_type = backbone_type
        self.augmentator = augmentator
        self.lut_map_dir = lut_map_dir
        self.lut_output_dir = lut_output_dir
        with open(val_list, 'r') as f:
            for line in f.readlines():
                tar_name, mask_name, cur_name = line.split()
                tar_name = tar_name.replace('\\', '/')
                mask_name = mask_name.replace('\\', '/')
                cur_name = cur_name.replace('\\', '/')
                cur_name = os.path.join(self.dataset_path, cur_name)
                mask_name = os.path.join(self.dataset_path, mask_name)
                tar_name = os.path.join(self.dataset_path, tar_name)
                self.tasks.append([tar_name, mask_name, cur_name])


    def __getitem__(self, index):
        sample = {}
        tar_name, mask_name, cur_name = self.tasks[index]


        cur_img = self.augmentator(image=cv2.imread(cur_name))["image"][:, :, ::-1].copy()
        cur_mask = cv2.cvtColor(cv2.imread(mask_name), cv2.COLOR_BGR2RGB)[:, :, 0].astype(np.float32) / 255.
        cur_mask = self.augmentator(object_mask=cur_mask, image=cv2.imread(tar_name))['object_mask']
        tar_img = self.augmentator(image=cv2.imread(tar_name))["image"][:, :, ::-1].copy()

        video, obj, img_number = cur_name.split('/')[-3:]
        lut_output_name = os.path.join(self.lut_output_dir, video + '_' + obj + '_' + img_number[:-4] + '.npy')
        lut_map_name = os.path.join(self.lut_map_dir, video + '_' + obj + '_' + img_number[:-4] + '.npy')
        assert os.path.exists(lut_output_name)
        assert os.path.exists(lut_map_name)

        lut_output = np.load(lut_output_name)
        lut_map = np.load(lut_map_name)
        cur_img = self.input_transform(cur_img)
        tar_img = self.input_transform(tar_img)
        sample['images'] = cur_img
        sample['masks'] = cur_mask[np.newaxis, ...].astype(np.float32)
        sample['target_images'] = tar_img
        sample['name'] = cur_name
        sample['lut_output'] = torch.from_numpy(lut_output)
        sample['lut_map'] = torch.from_numpy(lut_map)

        return sample

    def __len__(self):
        return len(self.tasks)



class MyPreviousSequenceDataset(BaseHDataset):
    def __init__(self, dataset_list, dataset_path, previous_num, future_num, **kwargs):
        super(MyPreviousSequenceDataset, self).__init__(**kwargs)
        self.dataset_path = dataset_path
        self.dataset_samples = []
        self.previous_num = previous_num
        self.future_num = future_num
        with open(dataset_list, 'r') as f:
            for line in f.readlines():
                real_img_name, cur_mask_name, cur_img_name = line.strip().split()
                cur_img_name = cur_img_name.replace('\\', '/')
                cur_mask_name = cur_mask_name.replace('\\', '/')
                real_img_name = real_img_name.replace('\\', '/')
                cur_img_name = os.path.join(self.dataset_path, cur_img_name)
                cur_mask_name = os.path.join(self.dataset_path, cur_mask_name)
                real_img_name = os.path.join(self.dataset_path, real_img_name)
                path, number = os.path.split(cur_img_name)
                mask_path, mask_number = os.path.split(cur_mask_name)
                pre_img_names = []
                pre_mask_names = []
                future_img_names = []
                future_mask_names = []
                for p in range(1, previous_num + 1):
                    pre_number = '%05d' % (int(number[:-4])-5 *p) + number[-4:]
                    pre_mask_number = '%05d' % (int(mask_number[:-4]) -5*p) + mask_number[-4:]
                    #print(pre_mask_number)
                    pre_img_name = os.path.join(path, pre_number)
                    pre_mask_name = os.path.join(mask_path, pre_mask_number)
                    if not os.path.exists(pre_mask_name):
                        if len(pre_img_names) > 0:
                            pre_img_name = pre_img_names[-1]
                            pre_mask_name = pre_mask_names[-1]
                        else:
                            pre_img_name = cur_img_name
                            pre_mask_name = cur_mask_name
                    pre_img_names.append(pre_img_name)
                    pre_mask_names.append(pre_mask_name)
                for p in range(1, future_num + 1):
                    future_number = '%05d' % (int(number[:-4]) + 5 *p) + number[-4:]
                    future_mask_number = '%05d' % (int(mask_number[:-4]) + 5*p) + mask_number[-4:]
                    #print(pre_mask_number)
                    future_img_name = os.path.join(path, future_number)
                    future_mask_name = os.path.join(mask_path, future_mask_number)
                    if not os.path.exists(future_mask_name):
                        #future_img_name = "no pic"
                        #future_mask_name = "no pic"
                        if len(future_mask_names) > 0:
                            future_img_name = future_img_names[-1]
                            future_mask_name = future_mask_names[-1]
                        else:
                            future_img_name = cur_img_name
                            future_mask_name = cur_mask_name
                    future_img_names.append(future_img_name)
                    future_mask_names.append(future_mask_name)
                self.dataset_samples.append((cur_img_name, cur_mask_name, pre_img_names, pre_mask_names, real_img_name, future_img_names, future_mask_names))

    def get_sample(self, index):
        cur_img_name, cur_mask_name, pre_img_names, pre_mask_names, real_img_name, future_img_names, future_mask_names = self.dataset_samples[index]
        cur_img = cv2.imread(cur_img_name)
        cur_img = cv2.cvtColor(cur_img, cv2.COLOR_BGR2RGB)

        real_img = cv2.imread(real_img_name)
        real_img = cv2.cvtColor(real_img, cv2.COLOR_BGR2RGB)

        cur_mask = cv2.imread(cur_mask_name)
        cur_mask = cv2.cvtColor(cur_mask, cv2.COLOR_BGR2RGB)
        pre_imgs = []
        future_imgs = []
        pre_masks = []
        future_masks = []
        for p in range(self.previous_num):
            pre_img_name = pre_img_names[p]
            pre_mask_name = pre_mask_names[p]
            if pre_img_name == "no pic":
                if len(pre_imgs) == 0:
                    pre_img = copy.copy(cur_img)
                    pre_mask = copy.copy(cur_mask)
                    pre_mask = pre_mask[:, :, 0].astype(np.float32) / 255.
                else:
                    pre_img = copy.copy(pre_imgs[-1])
                    pre_mask = copy.copy(pre_masks[-1])

            else:
                pre_img = cv2.imread(pre_img_name)
                pre_img = cv2.cvtColor(pre_img, cv2.COLOR_BGR2RGB)
                pre_mask = cv2.imread(pre_mask_name)
                pre_mask = pre_mask[:, :, 0].astype(np.float32) / 255.
                assert pre_img.shape[0]>0


            #print(pre_mask_name, pre_mask.shape, cur_mask.shape, len(pre_masks))


            pre_imgs.append(pre_img)
            pre_masks.append(pre_mask)

        for p in range(self.future_num):
            future_img_name = future_img_names[p]
            future_mask_name = future_mask_names[p]
            if future_img_name == "no pic":
                if len(future_imgs) == 0:
                    future_img = copy.copy(cur_img)
                    future_mask = copy.copy(cur_mask)
                    future_mask = future_mask[:, :, 0].astype(np.float32) / 255.
                else:
                    future_img = copy.copy(future_imgs[-1])
                    future_mask = copy.copy(future_masks[-1])

            else:
                future_img = cv2.imread(future_img_name)
                future_img = cv2.cvtColor(future_img, cv2.COLOR_BGR2RGB)
                future_mask = cv2.imread(future_mask_name)
                future_mask = future_mask[:, :, 0].astype(np.float32) / 255.
                assert future_img.shape[0]>0
            future_imgs.append(future_img)
            future_masks.append(future_mask)
        pre_imgs += future_imgs
        pre_masks += future_masks
        assert len(pre_imgs) == len(pre_masks)
        #print(pre_mask.dtype, pre_img.dtype, cur_img.dtype, pre_img_name)
        #pre_imgs = np.array(pre_imgs)
        #pre_masks = np.array(pre_masks)
        cur_mask = cur_mask[:, :, 0].astype(np.float32) / 255.


        return {
            'name': cur_img_name,
            'image': cur_img,
            'object_mask': cur_mask,
            'target_image': real_img,
            'image_id': index,
            'pre_image': pre_imgs,
            'pre_object_mask': pre_masks
        }



class MyPreviousSequenceDataset_future(BaseHDataset):
    def __init__(self, dataset_list, dataset_path, previous_num, future_num, **kwargs):
        super(MyPreviousSequenceDataset_future, self).__init__(**kwargs)
        self.dataset_path = dataset_path
        self.dataset_samples = []
        self.previous_num = previous_num
        self.future_num = future_num
        with open(dataset_list, 'r') as f:
            for line in f.readlines():
                real_img_name, cur_mask_name, cur_img_name = line.strip().split()
                cur_img_name = cur_img_name.replace('\\', '/')
                cur_mask_name = cur_mask_name.replace('\\', '/')
                real_img_name = real_img_name.replace('\\', '/')
                cur_img_name = os.path.join(self.dataset_path, cur_img_name)
                cur_mask_name = os.path.join(self.dataset_path, cur_mask_name)
                real_img_name = os.path.join(self.dataset_path, real_img_name)
                #path, number = os.path.split(cur_img_name)
                path, number = cur_img_name[:-9], cur_img_name[-9:]
                #mask_path, mask_number = os.path.split(cur_mask_name)
                mask_path, mask_number = cur_mask_name[:-9], cur_mask_name[-9:]
                pre_img_names = []
                pre_mask_names = []
                future_img_names = []
                future_mask_names = []
                for p in range(1, previous_num + 1):
                    pre_number = '%05d' % (int(number[:-4])-5 *p) + number[-4:]
                    pre_mask_number = '%05d' % (int(mask_number[:-4]) -5*p) + mask_number[-4:]
                    #print(pre_mask_number)
                    #pre_img_name = os.path.join(path, pre_number)
                    pre_img_name = path + pre_number
                    #pre_mask_name = os.path.join(mask_path, pre_mask_number)
                    pre_mask_name = mask_path + pre_mask_number
                    if not os.path.exists(pre_mask_name):
                        #pre_img_name = "no pic"
                        #pre_mask_name = "no pic"
                        if len(pre_img_names) > 0:
                            pre_img_name = pre_img_names[-1]
                            pre_mask_name = pre_mask_names[-1]
                        else:
                            pre_img_name = cur_img_name
                            pre_mask_name = cur_mask_name
                    pre_img_names.append(pre_img_name)
                    pre_mask_names.append(pre_mask_name)
                for p in range(1, future_num + 1):
                    future_number = '%05d' % (int(number[:-4]) + 5 *p) + number[-4:]
                    future_mask_number = '%05d' % (int(mask_number[:-4]) + 5*p) + mask_number[-4:]
                    #print(pre_mask_number)
                    #future_img_name = os.path.join(path, future_number)
                    future_img_name = path + future_number
                    #future_mask_name = os.path.join(mask_path, future_mask_number)
                    future_mask_name = mask_path + future_mask_number
                    if not os.path.exists(future_mask_name):
                        #future_img_name = "no pic"
                        #future_mask_name = "no pic"
                        if len(future_mask_names) > 0:
                            future_img_name = future_img_names[-1]
                            future_mask_name = future_mask_names[-1]
                        else:
                            future_img_name = cur_img_name
                            future_mask_name = cur_mask_name
                    future_img_names.append(future_img_name)
                    future_mask_names.append(future_mask_name)
                self.dataset_samples.append((cur_img_name, cur_mask_name, pre_img_names, pre_mask_names, real_img_name, future_img_names, future_mask_names))

    def get_sample(self, index):
        cur_img_name, cur_mask_name, pre_img_names, pre_mask_names, real_img_name, future_img_names, future_mask_names = self.dataset_samples[index]
        cur_img = cv2.imread(cur_img_name)
        cur_img = cv2.cvtColor(cur_img, cv2.COLOR_BGR2RGB)

        real_img = cv2.imread(real_img_name)
        real_img = cv2.cvtColor(real_img, cv2.COLOR_BGR2RGB)

        cur_mask = cv2.imread(cur_mask_name)
        cur_mask = cv2.cvtColor(cur_mask, cv2.COLOR_BGR2RGB)
        pre_imgs = []
        future_imgs = []
        pre_masks = []
        future_masks = []
        for p in range(self.previous_num):
            pre_img_name = pre_img_names[p]
            pre_mask_name = pre_mask_names[p]
            if pre_img_name == "no pic":
                """
                img_shape = cur_img.shape
                mask_shape = cur_mask.shape
                pre_img = np.zeros(img_shape)
                pre_mask = np.zeros(mask_shape)
                pre_img = pre_img.astype(np.uint8)
                pre_mask = pre_mask.astype(np.uint8)
                """
                if len(pre_imgs) == 0:
                    pre_img = copy.copy(cur_img)
                    pre_mask = copy.copy(cur_mask)
                    pre_mask = pre_mask[:, :, 0].astype(np.float32) / 255.
                else:
                    pre_img = copy.copy(pre_imgs[-1])
                    pre_mask = copy.copy(pre_masks[-1])

            else:
                pre_img = cv2.imread(pre_img_name)
                pre_img = cv2.cvtColor(pre_img, cv2.COLOR_BGR2RGB)
                pre_mask = cv2.imread(pre_mask_name)
                pre_mask = pre_mask[:, :, 0].astype(np.float32) / 255.
                assert pre_img.shape[0]>0


            #print(pre_mask_name, pre_mask.shape, cur_mask.shape, len(pre_masks))


            pre_imgs.append(pre_img)
            pre_masks.append(pre_mask)

        for p in range(self.future_num):
            future_img_name = future_img_names[p]
            future_mask_name = future_mask_names[p]
            if future_img_name == "no pic":
                """
                img_shape = cur_img.shape
                mask_shape = cur_mask.shape
                pre_img = np.zeros(img_shape)
                pre_mask = np.zeros(mask_shape)
                pre_img = pre_img.astype(np.uint8)
                pre_mask = pre_mask.astype(np.uint8)
                """
                if len(future_imgs) == 0:
                    future_img = copy.copy(cur_img)
                    future_mask = copy.copy(cur_mask)
                    future_mask = future_mask[:, :, 0].astype(np.float32) / 255.
                else:
                    future_img = copy.copy(future_imgs[-1])
                    future_mask = copy.copy(future_masks[-1])

            else:
                future_img = cv2.imread(future_img_name)
                future_img = cv2.cvtColor(future_img, cv2.COLOR_BGR2RGB)
                future_mask = cv2.imread(future_mask_name)
                future_mask = future_mask[:, :, 0].astype(np.float32) / 255.
                assert future_img.shape[0]>0
            future_imgs.append(future_img)
            future_masks.append(future_mask)
        pre_imgs += future_imgs
        pre_masks += future_masks
        assert len(pre_imgs) == len(pre_masks)
        #print(pre_mask.dtype, pre_img.dtype, cur_img.dtype, pre_img_name)
        #pre_imgs = np.array(pre_imgs)
        #pre_masks = np.array(pre_masks)
        cur_mask = cur_mask[:, :, 0].astype(np.float32) / 255.


        return {
            'name': cur_img_name,
            'image': cur_img,
            'object_mask': cur_mask,
            'target_image': real_img,
            'image_id': index,
            'pre_image': pre_imgs,
            'pre_object_mask': pre_masks
        }

class MyPreviousDataset(BaseHDataset):
    def __init__(self, dataset_list, dataset_path, **kwargs):
        super(MyPreviousDataset, self).__init__(**kwargs)
        self.dataset_path = dataset_path
        self.dataset_samples = []
        with open(dataset_list, 'r') as f:
            for line in f.readlines():
                real_img_name, cur_mask_name, cur_img_name = line.strip().split()
                cur_img_name = cur_img_name.replace('\\', '/')
                cur_mask_name = cur_mask_name.replace('\\', '/')
                real_img_name = real_img_name.replace('\\', '/')
                cur_img_name = os.path.join(self.dataset_path, cur_img_name)
                cur_mask_name = os.path.join(self.dataset_path, cur_mask_name)
                real_img_name = os.path.join(self.dataset_path, real_img_name)
                path, number = os.path.split(cur_img_name)
                mask_path, mask_number = os.path.split(cur_mask_name)
                pre_number = '%05d' % (int(number[:-4])-5) + number[-4:]
                pre_mask_number = '%05d' % (int(mask_number[:-4]) -5) + mask_number[-4:]
                pre_img_name = os.path.join(path, pre_number)
                pre_mask_name = os.path.join(mask_path, pre_mask_number)
                if not os.path.exists(pre_mask_name):
                    pre_img_name = "no pic"
                    pre_mask_name = "no pic"
                self.dataset_samples.append((cur_img_name, cur_mask_name, pre_img_name, pre_mask_name, real_img_name))

    def get_sample(self, index):
        cur_img_name, cur_mask_name, pre_img_name, pre_mask_name, real_img_name = self.dataset_samples[index]
        cur_img = cv2.imread(cur_img_name)
        cur_img = cv2.cvtColor(cur_img, cv2.COLOR_BGR2RGB)

        real_img = cv2.imread(real_img_name)
        real_img = cv2.cvtColor(real_img, cv2.COLOR_BGR2RGB)

        cur_mask = cv2.imread(cur_mask_name)
        #cur_mask = cv2.cvtColor(cur_mask, cv2.COLOR_BGR2RGB)

        if pre_img_name == "no pic":
            """
            img_shape = cur_img.shape
            mask_shape = cur_mask.shape
            pre_img = np.zeros(img_shape)
            pre_mask = np.zeros(mask_shape)
            pre_img = pre_img.astype(np.uint8)
            pre_mask = pre_mask.astype(np.uint8)
            """
            pre_img = copy.copy(cur_img)
            pre_mask = copy.copy(cur_mask)

        else:
            pre_img = cv2.imread(pre_img_name)
            pre_img = cv2.cvtColor(pre_img, cv2.COLOR_BGR2RGB)
            pre_mask = cv2.imread(pre_mask_name)
            assert pre_img.shape[0]>0

        #print(pre_mask.dtype, pre_img.dtype, cur_img.dtype, pre_img_name)

        cur_mask = cur_mask[:, :, 0].astype(np.float32) / 255.
        pre_mask = pre_mask[:, :, 0].astype(np.float32) / 255.

        return {
            'image': cur_img,
            'object_mask': cur_mask,
            'target_image': real_img,
            'image_id': index,
            'pre_image': pre_img,
            'pre_object_mask': pre_mask
        }

class ComposeDataset(BaseHDataset):
    def __init__(self, datasets, **kwargs):
        super(ComposeDataset, self).__init__(**kwargs)

        self._datasets = datasets
        self.dataset_samples = []
        for dataset_indx, dataset in enumerate(self._datasets):
            self.dataset_samples.extend([(dataset_indx, i) for i in range(len(dataset))])

    def get_sample(self, index):
        dataset_indx, sample_indx = self.dataset_samples[index]
        return self._datasets[dataset_indx].get_sample(sample_indx)
