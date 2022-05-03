# CO2Net
Official implementation of [Deep Video Harmonization  with Color Mapping Consistency](https://arxiv.org/abs/2205.00687)

## Prepare
- Linux
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN
- Clone this repo:
```bash
git clone https://github.com/bcmi/Video-Harmonization-Dataset-HYouTube.git
cd Video-Harmonization-Dataset-HYouTube
cd CO2Net
```
### Prepare Dataset
Download HYoutube from [link]

### Install cuda package
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

## Evaluate by our released model
```bash
python3  scripts/evaluate_model.py --gpu=0 --dataset_path <Your path to HYouTube> --val_list ./test_frames.txt --backbone ./final_models/issam_backbone.pth --previous_num 8 --future_num 8  --use_feature --checkpoint ./final_models/issam_final.pth
```
Or evaluate without refinement module

```bash
python3  scripts/evaluate_model.py --gpu=0 --dataset_path <Your path to HYouTube> --val_list ./test_frames.txt --backbone ./final_models/issam_backbone.pth --previous_num 8 --future_num 8 
```
Your can also use your own backbone or whole models. Please replace Arguments checkpoint/backbone by your own model. 

## Train your own model
We use a two-step training step, which means we firstly train backbone on HYoutube and then fix backbone and train refinement module. 
We provide code for two backbone: [iSSAM](https://openaccess.thecvf.com/content/WACV2021/papers/Sofiiuk_Foreground-Aware_Semantic_Representations_for_Image_Harmonization_WACV_2021_paper.pdf) [WACV2021] and [RainNet](https://openaccess.thecvf.com/content/CVPR2021/papers/Ling_Region-Aware_Adaptive_Instance_Normalization_for_Image_Harmonization_CVPR_2021_paper.pdf)[CVPR2021]. You can follow the same path of their repo to train backbone model ([iSSAM](https://github.com/saic-vul/image_harmonization) and [RainNet](https://github.com/junleen/RainNet)). We release iSSAM backbone here.

Your can directly train by 
```bash
python3  scripts/my_train.py --gpu=1 --dataset_path <Your path to HYouTube> --train_list ./train_list.txt --val_list ./test_frames.txt --backbone <Your backbone model> --backbone_type <Your backbone type, we provide 'issam' and 'rain' here> --previous_num 8 --future_num 8 --use_feature --normalize_inside --exp_name <exp name>
```
But since we adopt two stage traing strategy, we highly recommand your to calculate and store the result of Lut like 

```bash
python3  scripts/evaluate_model.py --gpu=0 --dataset_path <Your path to HYouTube> --val_list ./test_frames.txt --backbone <Your backbone model> --previous_num 8 --future_num 8 --write_lut_output <directory to store lut output> --write_lut_map <directory to store lut map> 
```
then you can use 

```bash
python3  scripts/my_train.py --gpu=1 --dataset_path  <Your path to HYouTube> --train_list ./train_list.txt --val_list ./test_frames.txt --backbone  <Your backbone model> --previous_num 8 --future_num 8 --use_feature --normalize_inside --exp_name <exp_name> --lut_map_dir <directory to store lut map> --lut_output_dir <directory to store lut output>
```

Then you can evaluate it by above instruction.

## Evaluate temporal consistency
we need you to download HYouTube_next from [link] and install Flownet2
### Prepare
Please follow command of [FlowNetV2](https://github.com/NVIDIA/flownet2-pytorch) to install and download FlowNetV2 weight.
### Prepare TL dataset
Please download TL dataset from [link]

### prepare result
You need to store the numpy result of model like 
```bash
python3  scripts/evaluate_model.py --gpu=0 --dataset_path <Your path to HYouTube> --val_list ./test_frames.txt --backbone <Your backbone model> --previous_num 8 --future_num 8 --checkpoint <Your checkpoint> --write_npy_result --result_npy_dir <Directory to store numpy result>
```
```bash
python3  scripts/evaluate_model.py --gpu=0 --dataset_path <Your path to HYouTube_Next> --val_list <next_frames.txt> --backbone <Your backbone model> --previous_num 8 --future_num 8 --checkpoint <Your checkpoint> --write_npy_result --result_npy_dir <Directory to store numpy future result>
```
Then calculate TL loss like
```bash
python3  scripts/evaluate_flow.py --dataset_path <Your path to HYouTube> --dataset_path_next <Your path to HYouTube_Next> --cur_result <result of current numpy dir> --next_result <result of next numpy dir>
```



