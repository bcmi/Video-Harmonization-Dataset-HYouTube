## Train baseline
We implement a simple version of [Temporally Coherent Video Harmonization Using Adversarial Networks](https://arxiv.org/pdf/1809.01372.pdf) using issam as backbone. You can train it by:
```bash
cd Huang et al
python3 train.py models/fixed256/improved_ssam.py --gpu=0 --worker=1 --dataset_path <Your path to HYouTube> --train_list ./train_frames.txt --val_list ./test_frames.txt
```
