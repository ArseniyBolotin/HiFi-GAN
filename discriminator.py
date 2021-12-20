import torch
import torch.nn.functional as F
import torch.nn as nn


class MPDBlock(torch.nn.Module):
    def __init__(self, period):
        super(MPDBlock, self).__init__()
        self.period = period
        self.layers = nn.ModuleList()
        channels = 1
        for l in range(1, 5):
            out_channels = 2 ** (5 + l)
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(channels, out_channels, kernel_size=(5, 1), stride=(3, 1), padding=(2, 0)),
                    nn.LeakyReLU(0.1)
                )
            )
            channels = out_channels
        self.layers.append(
            nn.Sequential(
                    nn.Conv2d(channels, 1024, kernel_size=(5, 1), padding=(2, 0)),
                    nn.LeakyReLU(0.1)
            )
        )
        self.layers.append(nn.Conv2d(1024, 1, kernel_size=(3, 1), padding=(1, 0)))

    def forward(self, x):
        batch_size = x.shape[0]
        features = []
        diff = x.shape[2] % self.period
        if diff != 0:
            x = F.pad(x, (0, self.period - diff))

        x = x.view(batch_size, -1, x.shape[2] // self.period, self.period)

        for layer in self.layers:
            x = layer(x)
            features.append(x)

        x = x.view(batch_size, -1)
        return x, features


class MPD(torch.nn.Module):
    def __init__(self):
        super(MPD, self).__init__()

        self.subdiscriminators = nn.ModuleList()
        for period in [2, 3, 5, 7, 11]:
            self.subdiscriminators.append(MPDBlock(period))

    def forward(self, y, y_preds):
        real_outputs = []
        pred_outputs = []
        real_features = []
        pred_features = []
        for d in self.subdiscriminators:
            d_output, features = d(y)
            real_outputs.append(d_output)
            real_features.append(features)
            d_output, features = d(y_preds)
            pred_outputs.append(d_output)
            pred_features.append(features)
        return real_outputs, pred_outputs, real_features, pred_features


class MSDBlock(torch.nn.Module):
    def __init__(self):
        super(MSDBlock, self).__init__()
        out_channels = [128, 128, 256, 512, 1024, 1024, 1024]
        kernel_sizes = [15, 41, 41, 41, 41, 41, 5]
        strides = [1, 2, 2, 4, 4, 1, 1]
        groups = [1, 4, 16, 16, 16, 16, 1]
        self.layers = nn.ModuleList()
        channels = 1
        for outs, k_sz, s, g in zip(out_channels, kernel_sizes, strides, groups):
            self.layers.append(
                nn.Sequential(
                    nn.Conv1d(channels, outs, kernel_size=k_sz, stride=s, groups=g, padding=k_sz // 2),
                    nn.LeakyReLU(0.1)
                )
            )
            channels = outs
        self.layers.append(nn.Conv1d(1024, 1, kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        batch_size = x.shape[0]
        features = []
        for layer in self.layers:
            x = layer(x)
            features.append(x)
        x = x.view(batch_size, -1)
        return x, features


class MSD(torch.nn.Module):
    def __init__(self, size=3):
        super(MSD, self).__init__()
        self.size = size
        self.subdiscriminators = nn.ModuleList()
        self.poolings = nn.ModuleList()
        for _ in range(size):
            self.subdiscriminators.append(MSDBlock())
        for _ in range(size - 1):
            self.poolings.append(nn.AvgPool1d(kernel_size=4, stride=2, padding=2))

    def forward(self, y, y_preds):
        real_outputs = []
        pred_outputs = []
        real_features = []
        pred_features = []
        for index in range(self.size):
            d = self.subdiscriminators[index]
            d_output, features = d(y)
            real_outputs.append(d_output)
            real_features.append(features)
            d_output, features = d(y_preds)
            pred_outputs.append(d_output)
            pred_features.append(features)
            if index > 0:
                y = self.poolings[index - 1](y)
                y_preds = self.poolings[index - 1](y_preds)
        return real_outputs, pred_outputs, real_features, pred_features
