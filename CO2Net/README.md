# CO2Net
Official implementation of [Deep Video Harmonization  with Color Mapping Consistency](https://arxiv.org/abs/2205.00687)

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
We provide two CUDA operation here for LUT calculation. Please make sure you have already installed CUDA. 
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
We use a two-step training step, which means we firstly train backbone on HYoutube and then fix backbone and train refinement module.

For stage 1: Backbone training, we provide code for two backbone: [iSSAM](https://openaccess.thecvf.com/content/WACV2021/papers/Sofiiuk_Foreground-Aware_Semantic_Representations_for_Image_Harmonization_WACV_2021_paper.pdf) [WACV2021] and [RainNet](https://openaccess.thecvf.com/content/CVPR2021/papers/Ling_Region-Aware_Adaptive_Instance_Normalization_for_Image_Harmonization_CVPR_2021_paper.pdf)[CVPR2021].You can follow the same path of their repo to train your own backbone model ([iSSAM](https://github.com/saic-vul/image_harmonization) and [RainNet](https://github.com/junleen/RainNet)). 
We release iSSAM backbone here (``` ./final_models/issam_backbone.pth ```).

For stage 2: Refinement module training, you can directly train by 
```bash
python3  scripts/my_train.py --gpu=1 --dataset_path <Your path to HYouTube> --train_list ./train_list.txt --val_list ./test_frames.txt --backbone <Your backbone model> --backbone_type <Your backbone type, we provide 'issam' and 'rain' here> --previous_num 8 --future_num 8 --use_feature --normalize_inside --exp_name <exp name>
```

But since we adopt two stage training strategy, we highly recommand you to calculate and store the result of LUT firstly using 
```bash
python3  scripts/evaluate_model.py --gpu=0 --dataset_path <Your path to HYouTube> --val_list ./all_frames.txt --backbone_type <Your backbone type> --backbone <Your backbone model> --previous_num 8 --future_num 8 --write_lut_output <directory to store lut output> --write_lut_map <directory to store lut map> 
```
then you can use 
```bash
python3  scripts/my_train.py --gpu=1 --dataset_path  <Your path to HYouTube> --train_list ./train_list.txt --val_list ./test_frames.txt --backbone_type <Your backbone type> --backbone  <Your backbone model> --previous_num 8 --future_num 8 --use_feature --normalize_inside --exp_name <exp_name> --lut_map_dir <directory to store lut map> --lut_output_dir <directory to store lut output>
```
It will directly read LUT result and not need to read all neigbors images. It will speed up.

Notice you can also choose your own previous number and future number of neigbors by changing Arguments **previous_num/future_num**. The Argument **Use_feature** decides whether to use final feature of backbone model. You can refer Table 2 in the paper for more information.



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

The expected quantitative results are as the following table. 
|      | MSE | FMSE | PSNR | fSSIM | 
| :--: | :---: | :------: | :-----: | :--------: | 
| Backbone  | 28.90 |  203.77   | 37.38  |   0.8817  |  
| Huang | 27.89 |  199.89   |  37.44  |   0.8821    | 
| Ours | 26.50 |  186.72   |  37.61  |   0.8827    |  

Your can also use your own backbone or whole models. Please replace Arguments **checkpoint/backbone** by your own model. 



## Evaluate temporal consistency
we need you to download TL test set, which is sub test set for calculating temporal loss (TL) and prepare Flownet2, which is used for calculating flow and image warping. TL test set is generated from FlowNet2 and the next unannotated frame of HYouTube. For more information, please see Section 3 in the supplementary.

### Prepare FlowNetV2
Please follow command of [FlowNet2](https://github.com/NVIDIA/flownet2-pytorch) to install and download FlowNet2 weight. Please put FlowNet directory on ``` ./ ``` and its weight on  ``` ./flownet/FlowNet2_checkpoint.pth.tar ```
### Prepare TL dataset
Please download TL test set from [**Baidu Cloud**](https://pan.baidu.com/s/1jpiPSkXoj_X3fWWk2vYCqw) (access code: 3v1s)

### prepare result
You need to store the both numpy result of candidate model on both HYoutube's test set and TL test set.
```bash
python3  scripts/evaluate_model.py --gpu=0 --dataset_path <Your path to HYouTube> --val_list ./test_frames.txt --backbone <Your backbone model> --previous_num 8 --future_num 8 --checkpoint <Your checkpoint> --write_npy_result --result_npy_dir <Directory to store numpy result>
```
```bash
python3  scripts/evaluate_model.py --gpu=0 --dataset_path <Your path to TL_TestSet> --val_list ./future_list.txt --backbone <Your backbone model> --previous_num 8 --future_num 8 --checkpoint <Your checkpoint> --write_npy_result --result_npy_dir <Directory to store numpy future result>
```
Also, to evaluate TL of backbone, you can store results of backbone using 
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

The expected quantitative results of released models are as the following table. 
|      | Tl |
| :--: | :---: |
| Backbone  | 6.48 | 
| Huang | 6.49 | 
| Ours | 5.11 |  





