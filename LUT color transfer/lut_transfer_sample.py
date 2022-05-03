import os
import numpy as np
import cv2
from PIL import Image
from pillow_lut import load_hald_image,rgb_color_enhance,load_cube_file
import random
import json


def get_f_b(real_img,mask,trans_img):
    real_img=np.array(real_img)
    mask=np.array(mask)
    trans_img=np.array(trans_img)

    mask = (1 - (mask == 0))
    mask = np.expand_dims(mask, axis=2)
    mask = mask.repeat(3, axis=2)

    out_img_np = mask * trans_img + (1 - mask) * real_img
    out_img = Image.fromarray(np.uint8(out_img_np))
    return out_img

def gen_fake(video_image_dir,video_object_mask_dir,video_object_fake_dir,random_lut):
    img_names=os.listdir(video_object_mask_dir)
    for img_name in img_names:
        real_img_path=os.path.join(video_image_dir,img_name)[:-3]+'jpg'
        mask_path=os.path.join(video_object_mask_dir,img_name)
        fake_img_path=os.path.join(video_object_fake_dir,img_name)[:-3]+'jpg'

        real_img=Image.open(real_img_path)
        mask=Image.open(mask_path)
        trans_img=real_img.filter(random_lut)

        fake_img=get_f_b(real_img,mask,trans_img)
        fake_img.save(fake_img_path)
    return

if __name__ == '__main__':
    video_image_dir='real_videos/003234408d'
    video_object_mask_dir='foreground_mask/003234408d/object_2'
    video_object_fake_dir='lut_sample'
    if not os.path.exists(video_object_fake_dir):
        os.makedirs(video_object_fake_dir)
    lut_path = 'LUTs/6.cube'
    hefe = load_cube_file(lut_path)
    lut = rgb_color_enhance(hefe, exposure=0.2, contrast=0.1, vibrance=0.5, gamma=1.3)
    gen_fake(video_image_dir,video_object_mask_dir,video_object_fake_dir,lut)




















