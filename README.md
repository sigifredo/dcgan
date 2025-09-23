# DCGAN (PyTorch) — 64×64

Implementación en **PyTorch** de un **DCGAN** clásico para generar imágenes a **64×64**. Incluye inicialización de pesos estilo Torch7, guardado de **checkpoints** y muestreo periódico a `samples/`.

---

## Características

- Arquitecturas **Generator / Discriminator** estilo DCGAN.
- Entrenamiento con Adam (`lr=2e-4`, `beta1=0.5` por defecto).
- Normalización de imágenes a `[-1, 1]` y salida `tanh` en G.
- Checkpoints por época (`checkpoints/*.pt`) y rejillas PNG (`samples/*.png`).

---

## Requisitos e instalación

### 1) Crear y activar entorno

**Linux / macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 2) Instalar dependencias

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install pillow numpy tqdm lmdb
```

---

## Preparación de datos

`ImageFolder` exige al menos una subcarpeta dentro de la raíz:

```bash
data/MyImages/imagefolder/
└── cualquier_nombre/     # p. ej. "mis_fotos"
    ├── img001.jpg
    ├── img002.png
    └── ...
```

**Nota:** no dejes imágenes directamente en `imagefolder` sin subcarpeta.

---

## Ejecutar entrenamiento

```bash
python3 train.py \
    --dataset folder \
    --data_root data/MyImages/imagefolder \
    --niter 25 \
    --batchSize 64 \
    --gpu 1 \
    --name dcgan_MyImages
```

Para ejecutar entrenamiento para 128:

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
