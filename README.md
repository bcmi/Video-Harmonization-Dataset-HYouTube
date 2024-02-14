# Video-Harmonization-Dataset-HYouTube

**Video composition** means cutting the foregrounds from one video and pasting them on the backgrounds from another video, resulting in a composite video. However, the inserted foregrounds may be incompatible with the backgrounds in terms of color and illumination statistics. **Video harmonization** aims to adjust the foregrounds in the composite video to make them compatible with the backgrounds, resulting in a harmonious video. Here are three examples of video harmonization, in which the top row contains composite videos and the bottom row contains the corresponding ground-truth harmonious videos.

![image](https://github.com/bcmi/Video-Harmonization-Dataset-HYouTube/blob/master/Example/2.gif)

## HYouTube Dataset

**Dataset Construction:** **Our dataset HYouTube is the first public video harmonization dataset** built upon large-scale video object segmentation dataset [Youtube-VOS-2018](https://youtube-vos.org/challenge/2018/). Given real videos with object masks, we adjust their foregrounds using Lookup Tables (LUTs) to produce synthetic composite videos. We employ in total 100 candidate LUTs, in which one LUT corresponds to one type of color transfer. 
Given a video sample, we first randomly select an LUT from 100 candidate LUTs to transfer the foreground of each frame. The transferred foregrounds and the original backgrounds form the composite frames, and the composite frames form composite video samples. We provide the script `lut_transfer_sample.py` to generate composite video based on real video, foreground mask, and LUT.
Our dataset includes 3194 pairs of synthetic composite video samples and real video samples, which are split to 2558 training pairs and 636 test pairs. Each video sample contains 20 consecutive frames with the foreground mask for each frame.  Our HYouTube dataset can be downloaded from [**Baidu Cloud**](https://pan.baidu.com/s/1LG15_3M4ISSyhRiVa6coig) (access code: dk07) or [**Bcmi Cloud**](https://cloud.bcmi.sjtu.edu.cn/sharing/MI8ygiNQZ). 

<img src='Example/dataset_construction.png' align="center" width=512>

**HYoutube File Structure:** Download the HYoutue dataset. We show the file structure below:

  ```
  ├── Composite: 
       ├── videoID: 
                ├── objectID
                          ├── imgID.jpg
                
                ├── ……
       ├── ……
  ├── Mask: 
       ├── videoID: 
                ├── objectID
                          ├── imgID.png
                
                ├── ……
       ├── ……
  ├── Ground-truth: 
       ├── videoID: 
                ├── imgID.jpg
       ├── ……
  ├── train.txt
  └── test.txt
  └── transfer.py
  ```

## Real Composite Videos

For evaluation, we also create 100 real composite videos.  Specifically, we collect  video  foregrounds  with  masks from  a  [video  matting  dataset](https://github.com/nowsyn/DVM)  as  well  as video backgrounds from [Vimeo-90k Dataset](http://toflow.csail.mit.edu/)  and  Internet.  Then,  we  create  composite  videos  via copy-and-paste and finally select 100 composite videos which look reasonable w.r.t. foreground  placement  but  inharmonious w.r.t. color/illumination.  100 real composite videos can be downloaded from [**Baidu Cloud**](https://pan.baidu.com/s/1ID7QDt1IkjT3plV1W-f8hg) (access code: nf9b) or [**Bcmi Cloud**](https://cloud.bcmi.sjtu.edu.cn/sharing/TyqpLmNNq).


# Color Transfer Method

#### Prerequisites
- Python 

- os

- numpy

- cv2

- PIL

- pillow_lut
#### Demo
We provide the script lut_transfer_sample.py to generate composite video based on real video, foreground mask, and LUT.

  ```
  python lut_transfer_sample.py
  ```
Before you run the code, you should change the path of real video directroy, the path of video mask directroy, the path of LUT and the storage path to generate your composite video.


# Our CO2Net
Official implementation of [Deep Video Harmonization  with Color Mapping Consistency](https://arxiv.org/pdf/2205.00687.pdf)

## Prepare
- Linux
- Python 3
- NVIDIA GPU + CUDA CuDNN
- Clone this repo:
```bash
git clone https://github.com/bcmi/Video-Harmonization-Dataset-HYouTube.git
cd Video-Harmonization-Dataset-HYouTube
cd CO2Net
```
### Prepare Dataset
Download HYoutube from [**Baidu Cloud**](https://pan.baidu.com/s/1LG15_3M4ISSyhRiVa6coig) (access code: dk07) or [**Bcmi Cloud**](https://cloud.bcmi.sjtu.edu.cn/sharing/MI8ygiNQZ). 

### Install cuda package
We provide two CUDA operations here for LUT calculation. Please make sure that you have already installed CUDA. 
```bash
cd CO2Net
cd trilinear
. ./setup.sh
```

```bash
cd CO2Net
cd tridistribute
. ./setup.sh
```
### Install python package
```bash
pip install -r requirements.txt
```

## Train 
We adopt a two-stage training strategy. In the first stage, we train an image harmonization backbone on HYoutube. In the second stage, we fix the backbone and train refinement module.

For stage 1: Backbone training, we provide the links for two backbones: [iSSAM](https://openaccess.thecvf.com/content/WACV2021/papers/Sofiiuk_Foreground-Aware_Semantic_Representations_for_Image_Harmonization_WACV_2021_paper.pdf) [WACV2021] and [RainNet](https://openaccess.thecvf.com/content/CVPR2021/papers/Ling_Region-Aware_Adaptive_Instance_Normalization_for_Image_Harmonization_CVPR_2021_paper.pdf)[CVPR2021].You can follow the same paths of their repos to train your own backbone model ([iSSAM](https://github.com/saic-vul/image_harmonization) and [RainNet](https://github.com/junleen/RainNet)). 
We release our trained iSSAM backbone here (``` ./final_models/issam_backbone.pth ```).

For stage 2: Refinement module training, you can directly train by 
```bash
python3  scripts/my_train.py --gpu=1 --dataset_path <Your path to HYouTube> --train_list ./train_list.txt --val_list ./test_frames.txt --backbone <Your backbone model> --backbone_type <Your backbone type, we provide 'issam' and 'rain' here> --previous_num 8 --future_num 8 --use_feature --normalize_inside --exp_name <exp name>
```

But since we adopt two-stage training strategy, we highly recommend you to calculate and store the LUT results first by using 
```bash
python3  scripts/evaluate_model.py --gpu=0 --dataset_path <Your path to HYouTube> --val_list ./all_frames.txt --backbone_type <Your backbone type> --backbone <Your backbone model> --previous_num 8 --future_num 8 --write_lut_output <directory to store lut output> --write_lut_map <directory to store lut map> 
```
then you can use 
```bash
python3  scripts/my_train.py --gpu=1 --dataset_path  <Your path to HYouTube> --train_list ./train_list.txt --val_list ./test_frames.txt --backbone_type <Your backbone type> --backbone  <Your backbone model> --previous_num 8 --future_num 8 --use_feature --normalize_inside --exp_name <exp_name> --lut_map_dir <directory to store lut map> --lut_output_dir <directory to store lut output>
```
It can directly read LUT result without the need to read all neigboring frames, which will speed up the training process.

Notice that you can also decide your own number of previous and future neigbors by changing Arguments **previous_num/future_num**. The Argument **Use_feature** decides whether to use final feature of backbone model. You can refer to Table 2 in the paper for more information.


## Evaluate
We release our backbone of iSSAM(```./final_models/issam_backbone.pth```), our framework's result with iSSAM as backbone(```./final_models/issam_final.pth```).  To compar with our method, we also use [Huang et al.](https://arxiv.org/abs/1809.01372)'s way to train iSSAM and release it in ```./final_models/issam_huang.pth```. Notice the architecture of obtained model by Huang et al.'s method is totally the same as iSSAM. So you can treat it as another checkpoint of backbone.

```bash
python3  scripts/evaluate_model.py --gpu=0 --dataset_path <Your path to HYouTube> --val_list ./test_frames.txt --backbone ./final_models/issam_backbone.pth --previous_num 8 --future_num 8  --use_feature --checkpoint ./final_models/issam_final.pth
```
Or evaluate without refinement module, it will test the result of LUT output. 

```bash
python3  scripts/evaluate_model.py --gpu=0 --dataset_path <Your path to HYouTube> --val_list ./test_frames.txt --backbone ./final_models/issam_backbone.pth --previous_num 8 --future_num 8 
```
To evaluate Huang's result, run 
```bash
python3  scripts/evaluate_model.py --gpu=0 --dataset_path <Your path to HYouTube> --val_list ./test_frames.txt --backbone ./final_models/issam_huang.pth --previous_num 1 --future_num 0
```
and see the metrics of backbone.

The expected quantitative results are in the following table. 
|      | MSE | FMSE | PSNR | fSSIM | 
| :--: | :---: | :------: | :-----: | :--------: | 
| Backbone  | 28.90 |  203.77   | 37.38  |   0.8817  |  
| Huang | 27.89 |  199.89   |  37.44  |   0.8821    | 
| Ours | 26.50 |  186.72   |  37.61  |   0.8827    |  

Your can also use your own backbone or whole models. Please replace Arguments **checkpoint/backbone** by your own model. 



## Evaluate temporal consistency
You need to download TL test set, which is a sub test set for calculating temporal loss (TL). You also need to prepare Flownet2, which is used for flow calculation and image warping. TL test set is generated from FlowNet2 and the next unannotated frame of HYouTube. For more information, please see Section 3 in the supplementary.

### Prepare FlowNetV2
Please follow command of [FlowNet2](https://github.com/NVIDIA/flownet2-pytorch) to install and download FlowNet2 weight. Please put FlowNet directory on ``` ./ ``` and its weight on  ``` ./flownet/FlowNet2_checkpoint.pth.tar ```
### Prepare TL dataset
Please download TL test set from [**Baidu Cloud**](https://pan.baidu.com/s/1jpiPSkXoj_X3fWWk2vYCqw) (access code: 3v1s)

### Prepare result numpy files and Evaluate
You need to store the both numpy result of candidate model on both HYoutube's test set and TL test set.
```bash
python3  scripts/evaluate_model.py --gpu=0 --dataset_path <Your path to HYouTube> --val_list ./test_frames.txt --backbone <Your backbone model> --previous_num 8 --future_num 8 --checkpoint <Your checkpoint> --write_npy_result --result_npy_dir <Directory to store numpy result>
```
```bash
python3  scripts/evaluate_model.py --gpu=0 --dataset_path <Your path to TL_TestSet> --val_list ./future_list.txt --backbone <Your backbone model> --previous_num 8 --future_num 8 --checkpoint <Your checkpoint> --write_npy_result --result_npy_dir <Directory to store numpy future result>
```
Also, to evaluate TL of backbone, you can store the results of backbone using 
```bash
python3  scripts/evaluate_model.py --gpu=0 --dataset_path <Your path to HYouTube> --val_list ./test_frames.txt --backbone <Your backbone model> --previous_num 8 --future_num 8 --checkpoint <Your checkpoint> --write_npy_backbone --backbone_npy_dir <Directory to store numpy result>
```
```bash
python3  scripts/evaluate_model.py --gpu=0 --dataset_path <Your path to TL_TestSet> --val_list ./future_list.txt --backbone <Your backbone model> --previous_num 8 --future_num 8 --checkpoint <Your checkpoint> --write_npy_result --result_npy_dir <Directory to store numpy future result>
```
Then calculate TL loss using 
```bash
python3  scripts/evaluate_flow.py --dataset_path <Your path to HYouTube> --dataset_path_next <Your path to HYouTube_Next> --cur_result <result of current numpy dir> --next_result <result of next numpy dir>
```

The expected quantitative results of released models are in the following table. 
|      | Tl |
| :--: | :---: |
| Backbone  | 6.48 | 
| Huang | 6.49 | 
| Ours | 5.11 |  




# Baselines

## Huang et al.
We implement a simple version of [Temporally Coherent Video Harmonization Using Adversarial Networks](https://arxiv.org/pdf/1809.01372.pdf) using iSSAM as backbone. You can train it by:
```bash
python3  scripts/evaluate_model.py --gpu=0 --dataset_path <Your path to HYouTube> --val_list ./all_frames.txt --backbone_type <Your backbone type> --backbone <Your backbone model> --previous_num 8 --future_num 8 --write_lut_output <directory to store lut output> --write_lut_map <directory to store lut map> 
cd ..
cd issam_huang
python3 train.py models/fixed256/improved_ssam.py --gpu=0 --worker=1 --dataset_path <Your path to HYouTube> --train_list ./train_frames.txt --val_list ./test_frames.txt
```

# Other Resources

+ [Image-Harmonization-Dataset-iHarmony4](https://github.com/bcmi/Image-Harmonization-Dataset-iHarmony4)
+ [Awesome-Image-Harmonization](https://github.com/bcmi/Awesome-Image-Harmonization)
+ [Awesome-Image-Composition](https://github.com/bcmi/Awesome-Image-Composition)


# Bibtex

If you find this work useful for your research, please cite our paper using the following BibTeX  [[arxiv](https://arxiv.org/pdf/2205.00687.pdf)]:

```
@article{hyoutube2021,
  title={Deep Video Harmonization with Color Mapping Consistency},
  author={Xinyuan Lu, Shengyuan Huang, Li Niu, Wenyan Cong, Liqing Zhang},
  journal={IJCAI},
  year={2022}
}
```

