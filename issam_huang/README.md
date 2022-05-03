## Train baseline
We implement a simple version of [Temporally Coherent Video Harmonization Using Adversarial Networks](https://arxiv.org/pdf/1809.01372.pdf) using issam as backbone. You can train it by:
```bash
python3  scripts/evaluate_model.py --gpu=0 --dataset_path <Your path to HYouTube> --val_list ./all_frames.txt --backbone_type <Your backbone type> --backbone <Your backbone model> --previous_num 8 --future_num 8 --write_lut_output <directory to store lut output> --write_lut_map <directory to store lut map> 
cd ..
cd issam_huang
python3 train.py models/fixed256/improved_ssam.py --gpu=0 --worker=1 --dataset_path <Your path to HYouTube> --train_list ./train_frames.txt --val_list ./test_frames.txt
```
