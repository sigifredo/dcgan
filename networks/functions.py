# from .generator import Generator

import torch
import types


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda:0')
    elif getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def _torch_load_smart(path: str, device, mode: str = 'auto'):
    '''
    Carga robusta con soporte para 'weights_only' en PyTorch recientes.
    mode: 'auto' | 'true' | 'false'
    '''

    try:
        if mode == 'true':
            return torch.load(path, map_location=device, weights_only=True)
        elif mode == 'false':
            return torch.load(path, map_location=device, weights_only=False)
        else:  # auto
            try:
                return torch.load(path, map_location=device, weights_only=False)
            except TypeError:
                return torch.load(path, map_location=device)
    except TypeError:
        return torch.load(path, map_location=device)


def load_checkpoint(checkpoint_path: str, nc: int, device: torch.device, mode: str = 'auto'):
    ckpt = _torch_load_smart(checkpoint_path, device, mode)

    if isinstance(ckpt, dict):
        # Inferir hyperparams desde el ckpt si están disponibles
        nz = (ckpt.get('opts', {}) or {}).get('nz', 100)
        ngf = (ckpt.get('opts', {}) or {}).get('ngf', 64)
        size = (ckpt.get('opts', {}) or {}).get('fineSize', 64)

        # Construir generador
        if size >= 128:
            from .generator128 import Generator128 as Generator
        else:
            from .generator64 import Generator64 as Generator

        G = Generator(nz=nz, ngf=ngf, nc=nc).to(device)
        G.eval()

        # Cargar pesos
        if 'netG' in ckpt:
            state = ckpt['netG']

        missing, unexpected = G.load_state_dict(state, strict=False)

        if missing:
            print('[WARN] Faltan claves en state_dict:', missing)
        if unexpected:
            print('[WARN] Claves inesperadas en state_dict:', unexpected)

        return types.SimpleNamespace(
            {
                'epoch': ckpt.get('epoch', 0),
                'netG': G,
                'opts': {
                    'nz': nz,
                    'ngf': ngf,
                    'size': size,
                },
            }
        )
    else:
        raise Exception('Error leyendo el diccionario del checkpoint')


def save_network(epoch_final, netG, netD, optimizerG, optimizerD, opts, path):
    torch.save(
        {
            'epoch': epoch_final,
            'netG': netG.state_dict(),
            'netD': netD.state_dict(),
            'optG': optimizerG.state_dict(),
            'optD': optimizerD.state_dict(),
            'opts': vars(opts),
            'seed': opts.manualSeed,
        },
        path,
    )
