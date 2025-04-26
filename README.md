# EMTAL
Pytorch implementation of "Transforming Vision Transformer: Towards Efficient Multi-Task Asynchronous Learning", accepted at 38th Conference on Neural Information Processing Systems (NeurIPS 24).


## Usage

### Install

- Clone this repo:
```
bash

git clone https://github.com/Yewen1486/EMTAL.git

cd EMTAL
```

- Create a conda virtual environment and activate it:
    

```
bash

conda create -n EMTAL python=3.7 -y

conda activate EMTAL
```
- Install `CUDA==11.3` following

 the [official installation instructions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)

- Install `PyTorch==1.11.0` and `torchvision==0.12.0` with `CUDA==11.3`:



```bash

pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

```

- Install `timm==0.6.5`:

```bash

pip install timm==0.6.5

```

- Install other requirements:

```bash

pip install -r requirements.txt

```

### Data preparation

- FGVC


You can follow [Erudite](https://github.com/PRIS-CV/An-Erudite-FGVC-Model) to download them.

We also provide the processed dataset in [Google Drive](https://drive.google.com/drive/folders/14S-FaNyVpH0bcE4JpaNu4a7yzvBXG7EX?usp=sharing).

- VTAB

You can follow [SSF](https://github.com/dongzelian/SSF) to download them.

### Pre-trained model preparation

We prepared the clusterd Vit-B/16 in [Google Drive](https://drive.google.com/drive/folders/14S-FaNyVpH0bcE4JpaNu4a7yzvBXG7EX?usp=sharing).

### Fine-tuning a pre-trained model via EMTAL

To fine-tune a pre-trained ViT model via `EMTAL` on MTL-FGVC, first set `--ckpt_kmeans` to the path of clustered model weights, set `--root_dir` to the root folder, then run:

```bash

bash train_scripts/vit/fgvc/EMTAL/train_EMTAL.sh

```


### Citation

If this project is helpful for you, you can cite our paper:

```

@inproceedings{ZHONG_2024_EMTAL,
 author = {Zhong, Hanwen and Chen, Jiaxin and Zhang, Yutong and Huang, Di and Wang, Yunhong},
 booktitle = {Advances in Neural Information Processing Systems},
 pages = {81130--81156},
 title = {Transforming Vision Transformer: Towards Efficient Multi-Task Asynchronous Learner},
 volume = {37},
 year = {2024}
}


```

### Acknowledgement

The code is built upon [timm](https://github.com/rwightman/pytorch-image-models). The processing of the dataset refers to [Erudite](https://github.com/PRIS-CV/An-Erudite-FGVC-Model) and [SSF](https://github.com/dongzelian/SSF).

