#!/usr/bin/env python3

'''
generate.py — Cargar un checkpoint de DCGAN y generar imágenes

Compatible con los checkpoints guardados por main.py:
- Formato completo: {'epoch','netG','netD','optG','optD','opts','seed', ...}
- O solo state_dict del generador (torch.save(netG.state_dict(), ...))

Uso (ejemplos):
  python generate.py --ckpt checkpoints/dcgan_64_epoch_010.pt --out_dir gen_64 --n 64 --size 64
  python generate.py --ckpt checkpoints/dcgan_128_epoch_030.pt --out_dir gen_128 --n 64 --size 128 --batch 32
  # Si el checkpoint tiene 'opts', muchos params se infieren solos (size/nz/ngf):
  python generate.py --ckpt checkpoints/dcgan_celeba_final.pt --out_dir gen

Requisitos:
  pip install torch torchvision pillow
'''


import argparse
import math
import networks as net
import pathlib
import torch
import torchvision as tv


def to_01(t):
    '''Escala de [-1,1] a [0,1] para guardar.'''
    return t.clamp(-1, 1).add_(1).div_(2)


def sample_noise(n, nz, device, dist='normal'):
    if dist == 'uniform':
        return torch.rand(n, nz, 1, 1, device=device) * 2 - 1

    return torch.randn(n, nz, 1, 1, device=device)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Generar imágenes desde un checkpoint de DCGAN')
    ap.add_argument('--ckpt', type=str, required=True, help='Ruta al checkpoint (.pt)')
    ap.add_argument('--out_dir', type=str, default='generated', help='Carpeta de salida')
    ap.add_argument('--n', type=int, default=64, help='Cantidad total de imágenes a generar')
    ap.add_argument('--batch', type=int, default=64, help='Batch de generación')
    ap.add_argument('--nc', type=int, default=3, help='Canales (3 = RGB)')
    ap.add_argument('--zdist', type=str, choices=['normal', 'uniform'], default='normal', help='Distribución del ruido')
    ap.add_argument('--seed', type=int, default=None, help='Semilla (opcional)')
    ap.add_argument('--grid', action='store_true', help='Guardar también un grid (grid.png)')
    ap.add_argument('--nrow', type=int, default=8, help='Columnas del grid si --grid')

    args = ap.parse_args()

    device = net.get_device()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Cargar checkpoint
    ckpt = net.load_checkpoint(args.ckpt, args.nc, device)

    # Generar en lotes
    n = args.n
    b = args.batch
    total_batches = math.ceil(n / b)
    saved_paths = []

    with torch.no_grad():
        for bi in range(total_batches):
            cur = b if (bi + 1) * b <= n else (n - bi * b)
            z = sample_noise(cur, ckpt.opts['nz'], device, dist=args.zdist)
            fake = ckpt.netG(z)  # [-1,1]
            fake01 = to_01(fake).cpu()

            # Guardar imágenes individuales
            for j in range(cur):
                idx = bi * b + j
                path = out_dir / f'img_{idx:05d}.png'
                tv.utils.save_image(fake01[j], path)
                saved_paths.append(path)

        # Grid opcional
        if args.grid:
            # Si n no es múltiplo de nrow, recorta para grid cuadrado decente
            grid_n = min(n, (n // args.nrow) * args.nrow) or min(n, args.nrow)
            # Re-generamos para el grid (o podríamos cargar de disco; esto es más simple)
            z = sample_noise(grid_n, ckpt.opts['nz'], device, dist=args.zdist)
            fake = ckpt.netG(z)
            fake01 = to_01(fake).cpu()
            grid = tv.utils.make_grid(fake01, nrow=args.nrow, padding=2)
            tv.utils.save_image(grid, out_dir / 'grid.png')

    print(f'[OK] Guardadas {len(saved_paths)} imágenes en: {out_dir}')
    if args.grid:
        print(f'[OK] Guardado grid: {out_dir/'grid.png'}')
    print(f'[INFO] Config -> size={ckpt.opts["size"]}, nz={ckpt.opts["nz"]}, ngf={ckpt.opts["ngf"]}, device={device}, zdist={args.zdist}')
