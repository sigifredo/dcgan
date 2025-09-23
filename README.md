# DCGAN (PyTorch) — 64×64 / 128×128

This repository is the result of adapting [Soumith Chintala's torch implementation](https://github.com/soumith/dcgan.torch), originally implemented in Lua and released under the BSD License.

The code has been reimplemented in Python/PyTorch and is distributed under the GNU General Public License (GPL).

In addition, the implementation extends the original functionality to support training at 128x128 resolution.

## Overview

PyTorch implementation of a classic DCGAN for generating 64×64 images. Includes Torch7-style weight initialization, checkpoint saving, and periodic sampling to `samples/`.

## Features

- Generator / Discriminator DCGAN-style architectures.
- Training with Adam (`lr=2e-4`, `beta1=0.5` by default).
- Image normalization to `[-1, 1]` and `tanh` output in G.
- Epoch checkpoints (`checkpoints/_.pt`) and PNG grids (`samples/_.png`).

## Requirements and Installation

### 1) Create and activate a virtual environment

**Linux / macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 2) Install dependencies

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install pillow numpy tqdm lmdb
```

## Data Preparation

`ImageFolder` requires at least one subdirectory inside the dataset root:

```bash
data/MyImages/imagefolder/
└── cualquier_nombre/     # p. ej. "mis_fotos"
    ├── img001.jpg
    ├── img002.png
    └── ...
```

**Note:** Do not place images directly inside `imagefolder` without a subdirectory.

## Run Training

For **64×64**:

```bash
python3 train.py \
    --dataset folder \
    --data_root data/MyImages/imagefolder \
    --niter 25 \
    --batchSize 64 \
    --gpu 1 \
    --name dcgan_MyImages
```

For **128×128**:

```bash
python3 train.py \
    --dataset folder \
    --data_root ./data/MyImages/ \
    --loadSize 160 \
    --fineSize 128 \
    --batchSize 64 \
    --nz 100 \
    --ngf 64 \
    --ndf 64 \
    --niter 75 \
    --lr 0.0002 \
    --beta1 0.5 \
    --name dcgan_MyImages
```

## License Notice

This work includes code from [Soumith Chintala's implementation](https://github.com/soumith/dcgan.torch), originally implemented in Lua and released under the BSD License. That code has been adapted and extended in this repository.

Unless otherwise noted, all modifications and new contributions are reimplemented in Python/PyTorch and distributed under the terms of the GNU General Public License (GPL).

This means:

- Original portions remain under their respective BSD License.
- The combined and modified work is distributed under the GPL.
