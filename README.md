# MobileIQA: Exploiting Mobile-level Diverse Opinion Network For No-Reference Image Quality Assessment Using Knowledge Distillation

---

:rocket:  :rocket: :rocket: **News:**
- ✅ **October, 2024**: We update the training and testing code.
- ✅ **August, 2024**: We created this repository.

[![paper](https://img.shields.io/badge/arXiv-Paper-green.svg)](https://arxiv.org/abs/2409.01212)
[![download](https://img.shields.io/github/downloads/chencn2020/MobileIQA/total.svg)](https://github.com/chencn2020/MobileIQA/releases)
[![Open issue](https://img.shields.io/github/issues/chencn2020/MobileIQA)](https://github.com/chencn2020/MobileIQA/issues)
[![Closed issue](https://img.shields.io/github/issues-closed/chencn2020/MobileIQA)](https://github.com/chencn2020/MobileIQA/issues)
![visitors](https://visitor-badge.glitch.me/badge?page_id=chencn2020/MobileIQA)
[![GitHub Stars](https://img.shields.io/github/stars/chencn2020/MobileIQA?style=social)](https://github.com/chencn2020/MobileIQA)

## Checklist

TBU

- [x] Code for MobileIQA
- [x] Code for training
- [] Code for testing
- [] Checkpoint

## Catalogue
1. [Introduction](#Introduction)
3. [Usage For Training](#Training)
4. [Usage For Testing](#Testing)
5. [Results](#Results)
6. [Citation](#Citation)
7. [Acknowledgement](#Acknowledgement)


## Introduction
<div id="Introduction"></div>


> With the rising demand for high-resolution (HR) images, No-Reference Image Quality Assessment (NR-IQA) gains more attention, as it can ecaluate image quality in real-time on mobile devices and enhance user experience. However, existing NR-IQA methods often resize or crop the HR images into small resolution, which leads to a loss of important details. And most of them are of high computational complexity, which hinders their application on mobile devices due to limited computational resources. To address these challenges, we propose MobileIQA, a novel approach that utilizes lightweight backbones to efficiently assess image quality while preserving image details through high-resolution input. MobileIQA employs the proposed multi-view attention learning (MAL) module to capture diverse opinions, simulating subjective opinions provided by different annotators during the dataset annotation process. The model uses a teacher model to guide the learning of a student model through knowledge distillation. This method significantly reduces computational complexity while maintaining high performance. Experiments demonstrate that MobileIQA outperforms novel IQA methods on evaluation metrics and computational efficiency.

## Usage For Training
<div id="Training"></div>

### Preparation

Run the following commands to create the environment:

```
conda create -n mobileiqa python=3.10
conda activate mobileiqa
pip install torch torchvision torchaudio
```


Then run the following command to install the other dependency:

```commandline
pip install -r requirements.txt
```

---

You can download the LIVEC, BID, SPAQ, KonIQ and UHDIQA datasets from the following download link.

|        Dataset        |                        Image Number                        |  Score Type  |                                  Download Link                                  |
|:---------------------:|:----------------------------------------------------------:|:------------:|:-------------------------------------------------------------------------------:|
|         LIVEC         |     1162 images taken on a variety of mobile devices.      |     MOS      |       <a href="https://live.ece.utexas.edu/research/ChallengeDB/index.html" target="_blank">Link</a>       |
|          BID          |                   586 real-blur images.                    |     MOS      | <a href="https://github.com/zwx8981/UNIQUE#link-to-download-the-bid-dataset" target="_blank">Link</a>      |
|         SPAQ          |          11,125 images from 66 smartphone images.          |     MOS      |                     <a href="https://github.com/h4nwei/SPAQ" target="_blank">Link</a>                      |
|         KonIQ         |  10,073 images selected from public multimedia resources.  |     MOS      |           <a href="http://database.mmsp-kn.de/koniq-10k-database.html" target="_blank">Link</a>           |
|         UHDIQA         |  6,073 4K images.  |     MOS      |           <a href="https://database.mmsp-kn.de/uhd-iqa-benchmark-database.html" target="_blank">Link</a>           |



### Training process

1. You should replace the dataset path in [dataset_info.json](./utils/dataset/dataset_info.json) to your own dataset path.
2. Run the following command to train the MobileVit-IQA (Please review the [train.py](train.py) for more options).
```commandline
HF_ENDPOINT=https://hf-mirror.com python3 -u train.py --gpu_id 1 --seed 3407 --batch_size 8 --dataset uhdiqa --loss MSE --model MobileVit_IQA --save_path ./Running_Test
```

### Distillation process

Run the following command to train the MobileNet-IQA with the guidance from the teacher model.
```commandline
HF_ENDPOINT=https://hf-mirror.com python3 -u train.py --gpu_id 1 --seed 3407 --batch_size 8 --dataset uhdiqa --loss MSE --save_path ./Running_Distill --teacher_pkl YOUR_TEACHER_PKL
```


## Usage For Testing
<div id="Inference"> </div>

TBU

<!-- Run the following command to get an image quality score for one image or images in a directory.

```commandline
python3 test.py --pkl_path path_to_pkl --image_path path_to_image
``` -->


## Citation
<div id="Citation"> </div>

If our work is useful to your research, we will be grateful for you to cite our paper:

```
@misc{chen2024mobileiqaexploitingmobileleveldiverse,
      title={MobileIQA: Exploiting Mobile-level Diverse Opinion Network For No-Reference Image Quality Assessment Using Knowledge Distillation}, 
      author={Zewen Chen and Sunhan Xu and Yun Zeng and Haochen Guo and Jian Guo and Shuai Liu and Juan Wang and Bing Li and Weiming Hu and Dehua Liu and Hesong Li},
      year={2024},
      eprint={2409.01212},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2409.01212}, 
}
```
## Acknowledgement
<div id="Acknowledgement"></div>

We sincerely thank the great work [HyperIQA](https://github.com/SSL92/hyperIQA) and [MANIQA](https://github.com/IIGROUP/MANIQA). 
The code structure is partly based on their open repositories.