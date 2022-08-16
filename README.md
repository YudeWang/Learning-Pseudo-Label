# Learning Pseudo Labels

The implementation of [Learning pseudo labels for semi-and-weakly supervised semantic segmentation](https://www.sciencedirect.com/science/article/pii/S003132032200406X). 

You can also access the code from [gitee](https://gitee.com/hibercraft/learning-pseudo-label).


## Abstract

In this paper, we aim to tackle semi-and-weakly supervised semantic segmentation (SWSSS), where many image-level classification labels and a few pixel-level annotations are available. We believe the most crucial point for solving SWSSS is to produce high-quality pseudo labels, and our method deals with it from two perspectives. Firstly, we introduce a class-aware cross entropy (CCE) loss for network training. Compared to conventional cross entropy loss, CCE loss encourages the model to distinguish concurrent classes only and simplifies the learning target of pseudo label generation. Secondly, we propose a progressive cross training (PCT) method to build cross supervision between two networks with a dynamic evaluation mechanism, which progressively introduces high-quality predictions as additional supervision for network training. Our method significantly improves the quality of generated pseudo labels in the regime with extremely limited annotations. Extensive experiments demonstrate that our approach outperforms state-of-the-art methods significantly.



## Installation

- Download the repository.

  ```shell
  git clone https://github.com/YudeWang/Learning-Pseudo-Label.git
  cd Learning-Pseudo-Label
  ```

- Create anaconda environment and install python dependencies.

  ```shell
  conda create -n semiweak python=3.8
  conda activate semiweak
  pip install -r requirements.txt
  ```

- Create softlink to your dataset. Make sure that the dataset can be accessed by `$your_dataset_path/VOCdevkit/VOC2012...`

  ```shell
  ln -s $your_dataset_path data
  ```

  

## Experiments

All the experiments of this work are placed in `.experiment/deeplabv3+_voc_swsss/`. Our approach is a two-stage method, which trains the network to generate pseudo labels (stage-1) firstly and then retrains another network for final prediction (stage-2). 

```shell
cd experiment/deeplabv3+_voc_swsss/
```

### Quick start

We provide a script `run.py` in experiment folder including both train & test & inference of two stages.

```shell
export CUDA_VISIBLE_DEVICES=0
python run.py
```

Suppose you want to run each step individually, please check the stage-1 configuration file `config.py` and stage-2 configuration file `config_retrain.py` firstly to meet your custom setting. And then run the corresponding python script for train/test/inference.

| Step                                                         | Command                   | Config file         |
| ------------------------------------------------------------ | ------------------------- | ------------------- |
| stage 1 - Train the model for pseudo label generation        | ```python train.py```     | `config.py`         |
| stage 1 - Evaluate pseudo label on val set (w/ image-level labels) | ```python test.py```      | `config.py`         |
| stage 1 - Generate pseudo label on trainaug set              | ```python inference.py``` | `config.py`         |
| stage 2 - Retrain another model                              | ```python retrain.py```   | `config_retrain.py` |
| stage 2 - Evaluate retrained model on val set (w/o image-level labels) | ```python retest.py```    | `config_retrain.py` |

Tips: follow the `run.py` to modify the experiment setting in `config.py` and `config_retrain.py`.

### Model Zoo

Here are some trained models file:

| Size of strongly labeled subset | HybridNet (eval w/ image-level labels)                       | PseudoNet (eval w/ image-level labels)                       | Retrained model (eval w/o image-level labels)                |
| ------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 92                              | 77.9% mIoU<br/>[Google Drive](https://drive.google.com/file/d/1vVv860IfFafBkVxQFERisBe32kPMYVDg/view?usp=sharing)/[Baidu Drive](https://pan.baidu.com/s/1aJhGZCHmYBnXCK8FYstVqw?pwd=72vy) | 78.3% mIoU<br/>[Google Drive](https://drive.google.com/file/d/1gUHRoN7-8j9X6XaR2iheFcZOHeRNN37P/view?usp=sharing)/[Baidu Drive](https://pan.baidu.com/s/1vMaQNl1Y4k2vFkuOWBHJJg?pwd=n4yf) | 76.2% mIoU<br/>[Google Drive](https://drive.google.com/file/d/13crEcIn7xrB7yNCHfu_EeDQwOi2YlHZj/view?usp=sharing)/[Baidu Drive](https://pan.baidu.com/s/1k7NUu7WGFsb6wiEHCunjng?pwd=i23v) |
| 183                             | 79.7% mIoU<br/>[Google Drive](https://drive.google.com/file/d/1xoppARFykwNMwtTzAMQp-N0d9JzcUgKM/view?usp=sharing)/[Baidu Drive](https://pan.baidu.com/s/1iWQwzlbwNc47VktMz8NnhA?pwd=uhq1) | 79.2% mIoU<br/>[Google Drive](https://drive.google.com/file/d/1IAya6Q4zW0yLkEowvm43hNqrUoaXB-Ef/view?usp=sharing)/[Baidu Drive](https://pan.baidu.com/s/1OkxPhmRQElVmE-8LdR3BZA?pwd=v5rv) | 77.6% mIoU<br/>[Google Drive](https://drive.google.com/file/d/1WR65lWDecgaDnJ_zYSaDwkRkBkIXtOyy/view?usp=sharing)/[Baidu Drive](https://pan.baidu.com/s/13xM88fn5sXmVo4Tw1zivSA?pwd=51bz) |
| 366                             | 81.7% mIoU<br/>[Google Drive](https://drive.google.com/file/d/1HRVMz-VpF4rBDFZyPsmBb5hOj9yFf86W/view?usp=sharing)/[Baidu Drive](https://pan.baidu.com/s/1MtOV7RCctXhW54aXHpbO1w?pwd=5csx) | 82.4% mIoU<br/>[Google Drive](https://drive.google.com/file/d/1oV7p_FoTE8yL34C4HQA6HPHIlUnQzqD9/view?usp=sharing)/[Baidu Drive](https://pan.baidu.com/s/1R6iGS-VSsQCGuecntPcKZA?pwd=tzgh) | 78.7% mIoU<br/>[Google Drive](https://drive.google.com/file/d/1h7-P0CwCHPwReuIU1pGXdmcFJyYM9Yrw/view?usp=sharing)/[Baidu Drive](https://pan.baidu.com/s/1x_lqrXAMnbdSqMdEdXHpdQ?pwd=sgcz) |
| 732                             | 83.7% mIoU<br/>[Google Drive](https://drive.google.com/file/d/1NsxpnWdFbC2kwS77sdvicUXmamkQQKwV/view?usp=sharing)/[Baidu Drive](https://pan.baidu.com/s/10inb2_FINxAXYvAKPr-suw?pwd=u5ws) | 83.9% mIoU<br/>[Google Drive](https://drive.google.com/file/d/16F54W-ipsKD90U47C2om8gMXKC1nysZE/view?usp=sharing)/[Baidu Drive](https://pan.baidu.com/s/13WxKRZwlg7UbNTLrnN0lgA?pwd=4c8m) | 79.9% mIoU<br/>[Google Drive](https://drive.google.com/file/d/1vACGVCbiZXCPekK56ngW5mF_xyhdkeX_/view?usp=sharing)/[Baidu Drive](https://pan.baidu.com/s/168NrWLut-gwVs0LMsfs7QQ?pwd=wer7) |
| 1464                            | 86.2% mIoU<br/>[Google Drive](https://drive.google.com/file/d/1a9mVVxenZlLPt0zICEpYhl2W9HiJbNk0/view?usp=sharing)/[Baidu Drive](https://pan.baidu.com/s/1RlPUtGBa_JoMzVNHEnwgcw?pwd=4kvf) | 86.2% mIoU<br/>[Google Drive](https://drive.google.com/file/d/1M7RqEqN7-xL2o6GCXbDxuQ3n7mG-6dpK/view?usp=sharing)/[Baidu Drive](https://pan.baidu.com/s/1yMaQBKWDilKxJ6D7PjputA?pwd=sygp) | 81.2% mIoU<br/>[Google Drive](https://drive.google.com/file/d/1Vj1fXoGQfvzV_yxa50oUFZWVjmPUTuV0/view?usp=sharing)/[Baidu Drive](https://pan.baidu.com/s/1ZFeaFSvHbqGP7NRRMh39jA?pwd=ua1x) |

## Citation

Please cite our paper if the code is helpful for your research.

> ```
> @article{wang2022learning,
>   title={Learning Pseudo Labels for Semi-and-weakly Supervised Semantic Segmentation},
>   author={Wang, Yude and Zhang, Jie and Kan, Meina and Shan, Shiguang},
>   journal={Pattern Recognition},
>   pages={108925},
>   year={2022},
>   publisher={Elsevier}
> }
> ```
