#!/bin/env python3

'''
DCGAN 64x64 — Traducción fiel de Torch7/Lua a PyTorch (Python).
- Arquitectura: igual a la original (ConvTranspose/Conv + BN + ReLU/LeakyReLU + Tanh/Sigmoid).
- Entrenamiento: mismo esquema de actualización (D con real+fake, luego G con etiquetas reales).
- Init de pesos: normal(0,0.02) y BN gamma~N(1,0.02), beta=0 (conv con bias=False donde corresponde).
- Opciones replicadas: dataset, batchSize, loadSize, fineSize, nz, ngf, ndf, nThreads, niter, lr, beta1, ntrain, display, gpu, name, noise, epoch_save_modulo.
- Salidas: checkpoints/ y samples/ (grids PNG). Probado en PyTorch >= 2.1, torchvision >= 0.16. En Colab suele venir preinstalado.
'''

import argparse
import networks as net
import pathlib
import random
import time
import torch
import torchvision as tv
import torchvision.transforms as T


# ------------------------------------------------------------
# Utils
# ------------------------------------------------------------
def set_seed(seed: int, cuda_deterministic: bool = False):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if cuda_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def weights_init_dcgan(m: torch.nn.Module):
    '''
    Imita el weights_init de Torch7:
    - Convs: N(0, 0.02)
    - BN:    gamma ~ N(1, 0.02), beta = 0
    - Conv layers que anteceden BN usan bias=False (como m:noBias() en Torch).
    '''

    classname = m.__class__.__name__

    if 'Conv' in classname:
        if getattr(m, 'weight', None) is not None:
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
        if getattr(m, 'bias', None) is not None:
            torch.nn.init.zeros_(m.bias)
    elif 'BatchNorm' in classname:
        if getattr(m, 'weight', None) is not None:
            torch.nn.init.normal_(m.weight, 1.0, 0.02)
        if getattr(m, 'bias', None) is not None:
            torch.nn.init.zeros_(m.bias)


# ------------------------------------------------------------
# Datos
# ------------------------------------------------------------
def build_transforms(loadSize: int, fineSize: int):
    # Torch7 hacía loadSize (96) -> center crop a 64
    return T.Compose(
        [
            T.Resize(loadSize),
            T.CenterCrop(fineSize),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )


def get_dataset(name: str, data_root: str, loadSize: int, fineSize: int, lsun_classes: str):
    name = name.lower()
    tfm = build_transforms(loadSize, fineSize)

    if name == 'folder' or name == 'imagenet':
        # Para 'imagenet', apunta data_root al directorio con subcarpetas de clases (como ImageFolder).
        return tv.datasets.ImageFolder(root=data_root, transform=tfm)

    elif name == 'lsun':
        # Nota: LSUN es pesado y su descarga puede fallar en Colab sin credenciales adecuadas.
        # lsun_classes puede ser 'bedroom_train,church_outdoor_train' separado por comas.
        classes = [c.strip() for c in lsun_classes.split(',') if c.strip()]
        if not classes:
            classes = ['bedroom_train']
        return tv.datasets.LSUN(root=data_root, classes=classes, transform=tfm)

    elif name == 'cifar10':
        return tv.datasets.CIFAR10(root=data_root, train=True, download=True, transform=tfm)

    elif name == 'fake':
        # Para pruebas rápidas
        return tv.datasets.FakeData(size=4096, image_size=(3, fineSize, fineSize), transform=tfm)

    else:
        raise ValueError(f'Dataset no soportado: {name}. Usa lsun / imagenet / folder / cifar10 / fake')


# ------------------------------------------------------------
# Entrenamiento
# ------------------------------------------------------------
def sample_noise(batch_size: int, nz: int, device, noise_type: str):
    if noise_type == 'uniform':
        return torch.rand(batch_size, nz, 1, 1, device=device) * 2.0 - 1.0

    # default: normal
    return torch.randn(batch_size, nz, 1, 1, device=device)


def save_sample_grid(tensor_bchw: torch.Tensor, out_dir: pathlib.Path, stem: str, nrow: int = 8):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Tus imágenes salen de G en [-1, 1]; las llevamos a [0, 1] manualmente.
    t = tensor_bchw.detach().clamp_(-1, 1).add_(1).div_(2)
    grid = tv.utils.make_grid(t, nrow=nrow, padding=2)  # no normalize/value_range
    out_path = out_dir / f'{stem}.png'
    tv.utils.save_image(grid, out_path)

    return out_path


def train(opts):
    device = torch.device(f'cuda:{opts.gpu-1}' if (opts.gpu > 0 and torch.cuda.is_available()) else 'cpu')

    print('Opciones:', vars(opts))
    print(f'Device: {device}')

    # Semilla
    if opts.manualSeed is None:
        opts.manualSeed = random.randint(1, 10000)

    print('Random Seed:', opts.manualSeed)
    set_seed(opts.manualSeed, cuda_deterministic=False)

    # Datos
    dataset = get_dataset(opts.dataset, opts.data_root, opts.loadSize, opts.fineSize, opts.lsun_classes)
    print(f'Dataset: {opts.dataset}  Tamaño: {len(dataset)}')
    loader = torch.utils.data.DataLoader(dataset, batch_size=opts.batchSize, shuffle=True, num_workers=opts.nThreads, pin_memory=True if device.type == 'cuda' else False, drop_last=True)

    # Modelos — usando la fábrica 64×64 (crea o reanuda)
    cfg = net.ModelConfig(
        nz=opts.nz,
        ngf=opts.ngf,
        ndf=opts.ndf,
        nc=3,
        lr=opts.lr,
        beta1=opts.beta1,
        device=device,
        image_size=opts.fineSize,
    )

    res = net.build_or_resume(
        cfg,
        resume_path=opts.resume,
        weights_only_mode=opts.weights_only,
        strict_g=bool(opts.resume_strict),
    )

    nz = opts.nz
    netG, netD = res.G, res.D
    criterion = res.criterion
    optimizerG = res.optG
    optimizerD = res.optD
    start_epoch = res.start_epoch  # 1 o ckpt.epoch+1

    if getattr(res, 'ckpt_meta', None) and opts.manualSeed is None and isinstance(res.ckpt_meta, dict):
        ck_seed = res.ckpt_meta.get('seed', None)

        if ck_seed is not None:
            print(f'[Resume] Usando seed del checkpoint: {ck_seed}')
            opts.manualSeed = int(ck_seed)
            set_seed(opts.manualSeed, cuda_deterministic=False)

    real_label = 1.0
    fake_label = 0.0

    # Ruido fijo para visualización (como noise_vis)
    fixed_noise = sample_noise(opts.batchSize, nz, device, opts.noise)

    # Directorios
    ckpt_dir = pathlib.Path('checkpoints')
    ckpt_dir.mkdir(exist_ok=True)
    samples_dir = pathlib.Path('samples')
    samples_dir.mkdir(exist_ok=True)

    # Entrenamiento
    global_step = 0
    for epoch in range(start_epoch, opts.niter + 1):
        epoch_start = time.perf_counter()
        seen = 0  # ejemplos vistos para respetar ntrain

        for i, (imgs, _) in enumerate(loader, start=1):
            if seen >= opts.ntrain:
                break

            bsz = imgs.size(0)
            seen += bsz
            imgs = imgs.to(device)

            # -----------------------------------------
            # (1) Update D: maximize log(D(x)) + log(1 - D(G(z)))
            # -----------------------------------------
            netD.train()
            optimizerD.zero_grad(set_to_none=True)

            # Real
            label_real = torch.full((bsz,), real_label, dtype=torch.float, device=device)
            out_real = netD(imgs)
            errD_real = criterion(out_real, label_real)
            errD_real.backward()

            # Fake
            noise = sample_noise(bsz, nz, device, opts.noise)
            fake = netG(noise)
            label_fake = torch.full((bsz,), fake_label, dtype=torch.float, device=device)
            out_fake = netD(fake.detach())
            errD_fake = criterion(out_fake, label_fake)
            errD_fake.backward()

            errD = errD_real + errD_fake
            optimizerD.step()

            # -----------------------------------------
            # (2) Update G: maximize log(D(G(z)))
            # (etiquetas reales para el generador)
            # -----------------------------------------
            optimizerG.zero_grad(set_to_none=True)
            label_gen = torch.full((bsz,), real_label, dtype=torch.float, device=device)
            out_gen = netD(fake)
            errG = criterion(out_gen, label_gen)
            errG.backward()
            optimizerG.step()

            global_step += 1

            # Display / logging
            if opts.display and (global_step % opts.display_freq == 0):
                with torch.no_grad():
                    fake_vis = netG(fixed_noise).detach().cpu()
                img_path = save_sample_grid(fake_vis, samples_dir, f'{opts.name}_e{epoch:03d}_it{global_step:08d}', nrow=8)
                print(f'[Vis] Guardada muestra: {img_path}')

            if (i % opts.log_every) == 0:
                print(f'Epoch [{epoch}/{opts.niter}]  Iter [{i:05d}]  ' f'Err_G: {errG.item():.4f}  Err_D: {errD.item():.4f}  ' f'seen: {seen}/{min(len(dataset), opts.ntrain)}')

        # Fin de epoch — guardar checkpoints (modulo)
        if (epoch % opts.epoch_save_modulo) == 0:
            net.save_network(epoch, netG, netD, optimizerG, optimizerD, opts, ckpt_dir / f'{opts.name}_epoch_{epoch:03d}.pt')
            # también una muestra fija por epoch
            with torch.no_grad():
                fake_vis = netG(fixed_noise).detach().cpu()
            save_sample_grid(fake_vis, samples_dir, f'{opts.name}_epoch_{epoch:03d}', nrow=8)

        print(f'Fin de epoch {epoch}/{opts.niter}  Tiempo: {time.perf_counter() - epoch_start:.2f}s')

    epoch_final = epoch if 'epoch' in locals() else opts.niter
    final_path = ckpt_dir / f'{opts.name}_final.pt'

    net.save_network(epoch_final, netG, netD, optimizerG, optimizerD, opts, final_path)

    print(f'[Checkpoint] Guardado final en: {final_path}')
    print('Entrenamiento finalizado.')


# ------------------------------------------------------------
# Argumentos (mapeo 1:1 con tu tabla 'opt' de Lua donde aplica)
# ------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument('--resume', type=str, default=None, help='Ruta a checkpoint .pt para reanudar (empieza en epoch = ckpt.epoch + 1)')
    p.add_argument('--weights_only', type=str, default='auto', choices=['auto', 'true', 'false'], help='Modo de torch.load: seguridad/compatibilidad (ver FutureWarning de PyTorch)')
    p.add_argument('--resume_strict', type=int, default=1, help='Cargar netG con strict=1/0 (útil si cambiaste claves)')

    p.add_argument('--dataset', type=str, default='folder', choices=['imagenet', 'lsun', 'folder', 'cifar10', 'fake'], help='Tipo de dataset')
    p.add_argument('--data_root', type=str, default='./data', help='Ruta raíz del dataset (para folder/imagenet/lsun). En cifar10/fake solo carpeta de cache.')
    p.add_argument('--lsun_classes', type=str, default='bedroom_train', help='Clases LSUN separadas por coma, p.ej: "bedroom_train,church_outdoor_train"')

    p.add_argument('--batchSize', type=int, default=64)
    p.add_argument('--loadSize', type=int, default=160)
    p.add_argument('--fineSize', type=int, default=128)

    p.add_argument('--nz', type=int, default=100, help='Dimensión del vector Z')
    p.add_argument('--ngf', type=int, default=64, help='# filtros iniciales del generador')
    p.add_argument('--ndf', type=int, default=64, help='# filtros iniciales del discriminador')

    p.add_argument('--nThreads', type=int, default=4, help='# workers del DataLoader')
    p.add_argument('--niter', type=int, default=25, help='# epochs')

    p.add_argument('--lr', type=float, default=0.0002)
    p.add_argument('--beta1', type=float, default=0.5)

    p.add_argument('--ntrain', type=int, default=10**12, help='# ejemplos por epoch (usa un valor finito para epochs rápidos)')

    p.add_argument('--display', type=int, default=1, help='0 desactiva muestras; !=0 activa guardado periódico')
    p.add_argument('--display_freq', type=int, default=10, help='Cada cuántas iteraciones guardar muestras')
    p.add_argument('--log_every', type=int, default=1, help='Cada cuántas iteraciones loguear')

    p.add_argument('--gpu', type=int, default=1, help='gpu=0 usa CPU; gpu=X usa CUDA device X (en Colab suele ser 1)')
    p.add_argument('--name', type=str, default='experiment1')
    p.add_argument('--noise', type=str, default='normal', choices=['uniform', 'normal'])

    p.add_argument('--epoch_save_modulo', type=int, default=1, help='Guardar checkpoint cada N epochs')
    p.add_argument('--manualSeed', type=int, default=None, help='Semilla; None = aleatoria')

    args = p.parse_args()  # <-- en Colab/Jupyter, sin CLI. Cambia a None si lo corres como script.
    # Comportamiento de display como en Torch: 0 -> False
    args.display = bool(args.display)
    return args


if __name__ == '__main__':
    opts = parse_args()
    # Asegura carpetas básicas
    pathlib.Path(opts.data_root).mkdir(parents=True, exist_ok=True)
    train(opts)
