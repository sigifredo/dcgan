import torch


class Generator(torch.nn.Module):
    def __init__(self, nz: int, ngf: int, nc: int = 3):
        super().__init__()
        self.main = torch.nn.Sequential(
            # Z -> (ngf*8) x 4 x 4
            torch.nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            torch.nn.BatchNorm2d(ngf * 8),
            torch.nn.ReLU(True),
            # (ngf*8) -> (ngf*4) x 8 x 8
            torch.nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(ngf * 4),
            torch.nn.ReLU(True),
            # (ngf*4) -> (ngf*2) x 16 x 16
            torch.nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(ngf * 2),
            torch.nn.ReLU(True),
            # (ngf*2) -> (ngf) x 32 x 32
            torch.nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(ngf),
            torch.nn.ReLU(True),
            # (ngf) -> (nc) x 64 x 64
            torch.nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=True),
            torch.nn.Tanh(),
        )

    def forward(self, z):
        return self.main(z)
