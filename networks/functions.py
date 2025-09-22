from .generator import Generator

import torch
import types


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda:0')
    elif getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def load_checkpoint(checkpoint_path: str, nc: int, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)

    if isinstance(ckpt, dict):
        # Inferir hyperparams desde el ckpt si están disponibles
        nz = (ckpt.get('opts', {}) or {}).get('nz', 100)
        ngf = (ckpt.get('opts', {}) or {}).get('ngf', 64)
        size = (ckpt.get('opts', {}) or {}).get('fineSize', 64)

        # Construir generador
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
