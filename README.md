# Video-Harmonization-Dataset-HYoutube
Video harmonization is to address the problem during video composition. Given a composite video, video harmonization will adjust the foreground appearance to make it compatible with the back-ground, resulting in a more realistic composite video.

Our dataset **HYoutube** is based on [Youtube-VOS-2018](https://youtube-vos.org/challenge/2018/). Given real videos with object masks, we first select the videos which meet our requirements and then adjust theforegrounds of these videos to produce synthetic compositevideos. Based on real video samples, we adjust the appearance of their  foregrounds  to  make  them  incompatible  with  back-grounds,  producing  synthetic  composite  videos. We employ LUT to adjust the foreground appearance for convenience. Since one LUT corresponds to one type of color transfer, we can ensure the diversity of the composite videos by applying different LUTs to video sam-ples. We collect more than 400 LUTs from the Internet andselect 100 candidate LUTs among them with the largest mutual difference. 

<img src='Examples/dataset_construct.jpg' align="center" width=1024>

Illustration of video harmonization task (blue ar-rows) and dataset construction process (red arrows):Given a video sample, we first select a LUT from 100 candidate LUTs randomly to transfer the foreground of each frame. The transferred foregrounds and the original backgrounds form the composite frames, and the composite frames form composite video samples.

Our dataset includes 3194 pairs of synthetic composite video samples and real video samples. Each video sample contains 20 consecutive frames with the foreground mask for each frame. The training (resp., test) set contains 2558 (resp.,636) videos. We provide the dataset and corresponding text in [**Baidu Cloud**] (). And we provide our transferring script here (to generate composite video by mask and real video).

Besides, we also synthesize  real  composite videos.  we  first  collect  30  video  foregrounds  with  masksfrom  a  video  matting  dataset  (Sun  et  al.  2021)  as  well  as30 video backgrounds from Vimeo-90k Dataset (Xue et al.2019)  and  Internet.  Then,  we  create  composite  videos  viacopy-and-paste and select 100 composite videos which look reasonable ***w.r.t.*** foreground  placement  but  inharmonious ***w.r.t.*** color/illumination.  We provide the dataset and corresponding text in [**Baidu Cloud**] ().






## Getting Started

### HYoutube File Structure
- Download the HYoutue dataset. We show the file structure below:

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
  
### Apply Transfer
#### Prerequisites

