import torch


class Discriminator128(torch.nn.Module):
    def __init__(self, ndf: int, nc: int = 3):
        super().__init__()
        self.main = torch.nn.Sequential(
            # (nc) x 128 x 128 -> (ndf) x 64 x 64
            torch.nn.Conv2d(nc, ndf, 4, 2, 1, bias=True),
            torch.nn.LeakyReLU(0.2, inplace=True),
            # (ndf) -> (ndf*2) x 32 x 32
            torch.nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(ndf * 2),
            torch.nn.LeakyReLU(0.2, inplace=True),
            # (ndf*2) -> (ndf*4) x 16 x 16
            torch.nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(ndf * 4),
            torch.nn.LeakyReLU(0.2, inplace=True),
            # (ndf*4) -> (ndf*8) x 8 x 8
            torch.nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(ndf * 8),
            torch.nn.LeakyReLU(0.2, inplace=True),
            # (ndf*8) -> (ndf*16) x 4 x 4
            torch.nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(ndf * 16),
            torch.nn.LeakyReLU(0.2, inplace=True),
            # (ndf*16) -> 1 x 1 x 1
            torch.nn.Conv2d(ndf * 16, 1, 4, 1, 0, bias=True),
            # torch.nn.Sigmoid(),
        )

    def forward(self, x):
        out = self.main(x)
        return out.view(-1)
