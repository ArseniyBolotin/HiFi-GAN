import math
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence


def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)


class ResBlock(nn.Module):
    def __init__(self, channels, k, D):  # k[n], D[n]
        super().__init__()
        self.inner_cycle = len(D[0])
        self.outer_cycle = len(D)
        self.blocks = nn.ModuleList()
        for i in range(self.outer_cycle):
            inner_block = []
            for j in range(self.inner_cycle):
                inner_block.append(nn.LeakyReLU(0.1, True))
                inner_block.append(nn.Conv1d(channels, channels, kernel_size=k, dilation=D[i][j], padding=get_padding(k,D[i][j])))
            self.blocks.append(nn.Sequential(*inner_block))

    def forward(self, x):
        for i in range(self.outer_cycle):
            x = x + self.blocks[i](x)
        return x


class MRF(nn.Module):
    def __init__(self, channels, k_r, D_r):
        super().__init__()
        self.res_blocks = nn.ModuleList()
        for k, D in zip(k_r, D_r):
            self.res_blocks.append(ResBlock(channels, k, D))

    def forward(self, x):
        out = torch.zeros_like(x)
        for res_block in self.res_blocks:
            out = out + res_block(x)
        return out


class GeneratorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, k, k_r, D_r):
        super().__init__()
        self.layers = nn.Sequential(
            nn.LeakyReLU(0.1, True),
            nn.ConvTranspose1d(in_channels, out_channels, kernel_size=k, stride=k // 2, padding=k // 4),
            MRF(out_channels, k_r, D_r)
        )

    def forward(self, x):
        return self.layers(x)


class Generator(nn.Module):
    def __init__(self, config, mel_spectrogram_config):
        super().__init__()
        self.layers = []
        channels = config.h_u
        self.layers.append(nn.Conv1d(80, channels, kernel_size=7, stride=1, padding=3))
        for k in config.k_u:
            self.layers.append(GeneratorBlock(channels, channels // 2, k, config.k_r, config.D_r))
            channels = channels // 2

        self.layers.append(nn.LeakyReLU(0.1, True))
        self.layers.append(nn.Conv1d(channels, 1, kernel_size=7, stride=1, padding=3))
        self.layers.append(nn.Tanh())
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.layers(x)
