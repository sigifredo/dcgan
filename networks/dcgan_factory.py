'''
Fábrica DCGAN 64x64 — crear desde cero o reanudar desde checkpoint.

Patrones:
- Factory Method: DCGANFactory64 crea (G, D, criterion, optimizadores) a partir de una config.
- Memento: funciones para cargar estado desde un checkpoint y devolver epoch inicial.
- Mini-Facade: build_or_resume() decide crear o reanudar en una sola llamada.
'''

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Any, Dict

import networks as net
import pathlib
import torch


# ---------------------------------------------------------------------
# Configuración del modelo (64x64) — solo parámetros necesarios aquí
# ---------------------------------------------------------------------
@dataclass
class ModelConfig:
    nz: int = 100
    ngf: int = 64
    ndf: int = 64
    nc: int = 3
    lr: float = 2e-4
    beta1: float = 0.5
    device: torch.device = torch.device('cpu')


# ---------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------
def weights_init_dcgan(m: torch.nn.Module):
    '''
    Init estilo DCGAN: Conv ~ N(0,0.02), BN gamma~N(1,0.02), beta=0.
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


def torch_load_smart(path: str, map_location, mode: str = 'auto') -> Any:
    '''
    Carga robusta con soporte 'weights_only' (PyTorch recientes) y fallback.

    mode: 'auto' | 'true' | 'false'
      - 'true'  -> torch.load(..., weights_only=True)  (seguro para state_dict)
      - 'false' -> torch.load(..., weights_only=False) (compatibilidad con checkpoints completos)
      - 'auto'  -> intenta 'false' y hace fallback si la versión no soporta el argumento
    '''

    try:
        if mode == 'true':
            return torch.load(path, map_location=map_location, weights_only=True)
        elif mode == 'false':
            return torch.load(path, map_location=map_location, weights_only=False)
        else:
            try:
                return torch.load(path, map_location=map_location, weights_only=False)
            except TypeError:
                return torch.load(path, map_location=map_location)
    except TypeError:
        return torch.load(path, map_location=map_location)


# ---------------------------------------------------------------------
# Resultado empaquetado (para pasar todo al script principal)
# ---------------------------------------------------------------------
@dataclass
class BuildResult:
    G: torch.nn.Module
    D: torch.nn.Module
    criterion: torch.nn.Module
    optG: torch.optim.Optimizer
    optD: torch.optim.Optimizer
    start_epoch: int  # 1 si es desde cero; ckpt['epoch']+1 si reanudas
    ckpt_meta: Optional[Dict] = None  # opcional: dict del checkpoint si se cargó


# ---------------------------------------------------------------------
# Carga de checkpoint en modelos/optimizadores (Memento)
# ---------------------------------------------------------------------
def load_checkpoint_into_models(
    resume_path: pathlib.Path,
    device: torch.device,
    G: torch.nn.Module,
    D: torch.nn.Module,
    optG: torch.optim.Optimizer,
    optD: torch.optim.Optimizer,
    weights_only_mode: str = 'auto',
    strict_g: bool = True,
) -> Tuple[int, Optional[Dict]]:
    '''
    Carga checkpoint:
    - 'Completo' (con 'netG' y opcionalmente 'netD','optG','optD','epoch'): start_epoch = epoch+1
    - 'Solo state_dict de G': carga G; D/opt quedan tal cual; start_epoch = 1
    '''

    ckpt = torch_load_smart(str(resume_path), map_location=device, mode=weights_only_mode)

    if isinstance(ckpt, dict) and 'netG' in ckpt:
        # checkpoint completo
        miss_g, unexp_g = G.load_state_dict(ckpt['netG'], strict=strict_g)

        if miss_g:
            print('[Resume][G] faltan:', miss_g)
        if unexp_g:
            print('[Resume][G] inesperadas:', unexp_g)

        if 'netD' in ckpt:
            miss_d, unexp_d = D.load_state_dict(ckpt['netD'], strict=False)

            if miss_d:
                print('[Resume][D] faltan:', miss_d)
            if unexp_d:
                print('[Resume][D] inesperadas:', unexp_d)

        if 'optG' in ckpt:
            try:
                optG.load_state_dict(ckpt['optG'])
            except Exception as e:
                print(f'[Resume][optG] no cargado: {e}')

        if 'optD' in ckpt:
            try:
                optD.load_state_dict(ckpt['optD'])
            except Exception as e:
                print(f'[Resume][optD] no cargado: {e}')

        start_epoch = int(ckpt.get('epoch', 0)) + 1
        return start_epoch, ckpt

    # probablemente solo state_dict del generador
    miss_g, unexp_g = G.load_state_dict(ckpt, strict=False)

    if miss_g:
        print('[Resume][G-only] faltan:', miss_g)
    if unexp_g:
        print('[Resume][G-only] inesperadas:', unexp_g)

    return 1, None


# ---------------------------------------------------------------------
# Fábrica (Factory Method) para DCGAN 64x64
# ---------------------------------------------------------------------
class DCGANFactory64:
    @staticmethod
    def create_fresh(cfg: ModelConfig) -> BuildResult:
        '''
        Crea G/D/criterio/optimizadores desde cero (64x64) e inicializa pesos.
        '''

        G = net.Generator(nz=cfg.nz, ngf=cfg.ngf, nc=cfg.nc).to(cfg.device)
        D = net.Discriminator(ndf=cfg.ndf, nc=cfg.nc).to(cfg.device)
        G.apply(weights_init_dcgan)
        D.apply(weights_init_dcgan)

        # Pérdida y optimizadores (Sigmoid + BCELoss como en Torch7)
        criterion = torch.nn.BCELoss().to(cfg.device)
        optD = torch.optim.Adam(D.parameters(), lr=cfg.lr, betas=(cfg.beta1, 0.999))
        optG = torch.optim.Adam(G.parameters(), lr=cfg.lr, betas=(cfg.beta1, 0.999))

        return BuildResult(
            G=G,
            D=D,
            criterion=criterion,
            optG=optG,
            optD=optD,
            start_epoch=1,
            ckpt_meta=None,
        )

    @staticmethod
    def create_from_checkpoint(
        cfg: ModelConfig,
        ckpt_path: str | pathlib.Path,
        *,
        weights_only_mode: str = 'auto',
        strict_g: bool = True,
    ) -> BuildResult:
        '''
        Construye modelos/opt y los llena con un checkpoint (reanuda).
        '''

        ckpt_path = pathlib.Path(ckpt_path)

        if not ckpt_path.exists():
            raise FileNotFoundError(f'Checkpoint no encontrado: {ckpt_path}')

        # Primero crea 'en blanco'
        fresh = DCGANFactory64.create_fresh(cfg)

        # Luego carga estado
        start_epoch, meta = load_checkpoint_into_models(ckpt_path, cfg.device, fresh.G, fresh.D, fresh.optG, fresh.optD, weights_only_mode=weights_only_mode, strict_g=strict_g)
        fresh.start_epoch = start_epoch
        fresh.ckpt_meta = meta

        return fresh


# ---------------------------------------------------------------------
# Mini-Facade: una sola función para crear o reanudar
# ---------------------------------------------------------------------
def build_or_resume(
    cfg: ModelConfig,
    *,
    resume_path: Optional[str] = None,
    weights_only_mode: str = 'auto',
    strict_g: bool = True,
) -> BuildResult:
    '''
    Crea todo desde cero si resume_path=None.
    Si se pasa resume_path, construye y reanuda desde ese checkpoint.
    '''

    if resume_path is None:
        return DCGANFactory64.create_fresh(cfg)
    return DCGANFactory64.create_from_checkpoint(cfg, resume_path, weights_only_mode=weights_only_mode, strict_g=strict_g)
