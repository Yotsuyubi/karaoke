import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.conv import Conv1d
from .utils import MultiScaleSpectrogram


class DownsamplingBlock(nn.Module):

    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv = nn.Conv1d(in_channel, out_channel, 15)
        self.downsampling = nn.Conv1d(out_channel, out_channel, 15, stride=2)
        self.batchnorm1 = nn.BatchNorm1d(out_channel)
        self.batchnorm2 = nn.BatchNorm1d(out_channel)

    def forward(self, x):
        skip = nn.Tanh()(self.batchnorm1(self.conv(x)))
        return nn.Tanh()(self.batchnorm2(self.downsampling(skip))), skip


class UpsamplingBlock(nn.Module):

    def __init__(self, in_channel, middle_channel, out_channel):
        super().__init__()
        self.upsampling = nn.ConvTranspose1d(
            in_channel, in_channel, 15, stride=2)
        self.conv = nn.Conv1d(middle_channel, out_channel, 15)
        self.batchnorm1 = nn.BatchNorm1d(middle_channel)
        self.batchnorm2 = nn.BatchNorm1d(out_channel)

    def forward(self, x, skip):
        x = self.upsampling(x)
        if skip.size()[-1]-x.size()[-1] != 0:
            skip = skip[
                :, :,
                (skip.size()[-1]-x.size()[-1])//2:-(skip.size()[-1]-x.size()[-1])//2
            ]
        x = nn.Tanh()(self.batchnorm1(th.cat([x, skip], dim=1)))
        return nn.Tanh()(self.batchnorm2(self.conv(x)))


class WaveUNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.downsample1 = DownsamplingBlock(2, 32)
        self.downsample2 = DownsamplingBlock(32, 64)
        self.downsample3 = DownsamplingBlock(64, 128)
        self.downsample4 = DownsamplingBlock(128, 256)
        self.downsample5 = DownsamplingBlock(256, 512)

        self.conv = nn.Conv1d(512, 512, 3, padding=1)
        self.batchnorm = nn.BatchNorm1d(512)

        self.upsample1 = UpsamplingBlock(32, 32*2, 16)
        self.upsample2 = UpsamplingBlock(64, 64*2, 32)
        self.upsample3 = UpsamplingBlock(128, 128*2, 64)
        self.upsample4 = UpsamplingBlock(256, 256*2, 128)
        self.upsample5 = UpsamplingBlock(512, 512*2, 256)

        self.out_upsamplilng = nn.ConvTranspose1d(16, 16, 15, stride=2)
        self.out = nn.Conv1d(18, 4, 1)

    def forward(self, x):
        skip = x
        x, skip1 = self.downsample1(x)
        x, skip2 = self.downsample2(x)
        x, skip3 = self.downsample3(x)
        x, skip4 = self.downsample4(x)
        x, skip5 = self.downsample5(x)

        x = nn.Tanh()(self.batchnorm(self.conv(x)))

        x = self.upsample5(x, skip5)
        x = self.upsample4(x, skip4)
        x = self.upsample3(x, skip3)
        x = self.upsample2(x, skip2)
        x = self.upsample1(x, skip1)
        x = self.out_upsamplilng(x)
        if x.size()[-1]-skip.size()[-1] != 0:
            x = x[
                :, :,
                (x.size()[-1]-skip.size()[-1])//2:-(x.size()[-1]-skip.size()[-1])//2
            ]
        x = th.cat([skip, x], dim=1)

        return nn.Tanh()(self.out(x))


class PostNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv1d(4, 128, 1),
            nn.Tanh(),
            *[
                nn.Sequential(
                    nn.Conv1d(128, 128, 33, padding=16),
                    nn.Tanh()
                ) for _ in range(8)
            ],
            nn.Conv1d(128, 4, 1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.seq(x)


class WaveDiscriminator(nn.Module):

    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv1d(4, 128, 15),
            nn.LeakyReLU(),
            *[
                nn.Sequential(
                    nn.Conv1d(128, 128, 33, stride=4),
                    nn.Tanh()
                ) for _ in range(4)
            ],
            nn.Conv1d(128, 64, 3),
            nn.LeakyReLU(),
            nn.Conv1d(64, 1, 1),
            nn.AdaptiveAvgPool1d((1,)),
            nn.Flatten(),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.seq(x)


class SpectrogramDiscriminator(nn.Module):

    def __init__(self):
        super().__init__()
        self.in_conv = nn.Sequential(
            nn.Conv2d(4, 64, 15),
            nn.BatchNorm2d(64)
        )
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(64, 64, 1),
                nn.Sigmoid()
            ) for _ in range(4)
        ])
        self.features = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(64, 64, 9, padding=4),
            ) for _ in range(4)
        ])
        self.downsamples = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(64, 64, 9, padding=4, stride=2),
            ) for _ in range(4)
        ])
        self.out_conv = nn.Sequential(
            nn.Conv2d(64, 1, 1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.in_conv(x)
        for gate, feature, downsample in zip(self.gates, self.features, self.downsamples):
            x = downsample(gate(x) * feature(x))
        return self.out_conv(x)


if __name__ == "__main__":

    dummy = th.rand(1, 4, 44100*5)
    # print(PostNet()(WaveUNet()(dummy)).shape)
    print(WaveDiscriminator()(dummy).shape)

    specs = MultiScaleSpectrogram()(dummy)
    disc = SpectrogramDiscriminator()
    for spec in specs:
        print(disc(spec))
