import torch as th
import torch.nn as nn
from torchaudio.functional import equalizer_biquad
import torchaudio
import functools
import librosa
import random
from scipy import signal
import glob
from os import path


class IR():

    def __init__(self, root):
        self.filenames = glob.glob(path.join(root, "*.wav"))

    def __call__(self, post_process=None, wet_rate=1):
        ir, _ = torchaudio.load(random.choice(self.filenames))
        if post_process:
            ir = post_process(ir)
        return Convoluve(ir, wet_rate)


class Convoluve(nn.Module):

    def __init__(self, ir, wet_rate):
        super().__init__()
        self.ir = ir.detach().numpy()
        self.wet_rate = wet_rate

    def forward(self, x):
        dry = x.detach().numpy()
        wet = signal.convolve(dry, self.ir, mode="same")
        return th.tensor(dry*(1-self.wet_rate) + wet*self.wet_rate).type_as(x)


class RandomCrop(nn.Module):

    def __init__(self, sample_length):
        super().__init__()
        self.sample_length = sample_length

    def forward(self, x):
        x_length = x.size()[-1]
        begin = random.randint(0, x_length-self.sample_length)
        return x[:, begin:begin+self.sample_length]


class EQ(nn.Module):

    def __init__(self, gain, freq, Q=0.707, sample_rate=44100):
        super().__init__()
        self.eq = functools.partial(
            equalizer_biquad,
            sample_rate=sample_rate,
            center_freq=freq,
            gain=gain,
            Q=Q
        )

    def forward(self, x: th.Tensor):
        return self.eq(x)


class ChannelSwap(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return th.roll(x, 1, -2)


class SignSwap(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x: th.Tensor):
        x[:, :] *= -1.0
        return x


class Shifter(nn.Module):

    def __init__(self, sec, sample_rate=44100):
        super().__init__()
        self.shifter = functools.partial(
            th.roll,
            shifts=int(sec*sample_rate),
            dims=-1
        )

    def forward(self, x: th.Tensor):
        return self.shifter(x)


class Fader(nn.Module):

    def __init__(self, gain):
        super().__init__()

        def fader(x, gain):
            return x * gain

        self.fader = functools.partial(
            fader, gain=gain
        )

    def forward(self, x: th.Tensor):
        return self.fader(x)


class TimeStretch(nn.Module):

    def __init__(self, rate):
        super().__init__()
        self.time_stretch = functools.partial(
            librosa.effects.time_stretch,
            rate=rate,
        )
        self.rate = rate

    def forward(self, x: th.Tensor):
        x_numpy = x.detach().numpy()
        len_x = x.size()[-1]
        len_stretch = int(round(len_x / self.rate))
        x_new = th.zeros(x.size()).type_as(x)
        for n in range(2):
            stretch = th.tensor(self.time_stretch(x_numpy[n, :]))
            if self.rate < 1.0:
                x_new[n, :] += stretch[:len_x]
            else:
                x_new[n, :] += th.nn.ConstantPad1d(
                    (0, len_x - len_stretch), 0.0)(stretch)
        return x_new


class PitchShift(nn.Module):

    def __init__(self, semitone):
        super().__init__()
        self.pitch_shift = functools.partial(
            librosa.effects.pitch_shift,
            sr=44100, n_steps=semitone
        )

    def forward(self, x):
        x_numpy = x.detach().numpy()
        x_new = th.zeros(x.size()).type_as(x)
        for n in range(2):
            x_new[n, :] += th.tensor(self.pitch_shift(x_numpy[n, :]))
        return x_new
