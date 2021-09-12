import torch as th
from torch import functional
import torch.nn as nn
from torchaudio import functional as F
from torchaudio.transforms import Spectrogram


class MelScale(nn.Module):
    __constants__ = ['n_mels', 'sample_rate', 'f_min', 'f_max']

    def __init__(self,
                 n_mels: int = 128,
                 sample_rate: int = 16000,
                 f_min: float = 0.,
                 f_max=None,
                 n_stft=None,
                 norm=None,
                 mel_scale: str = "htk") -> None:
        super(MelScale, self).__init__()
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.f_max = f_max if f_max is not None else float(sample_rate // 2)
        self.f_min = f_min
        self.norm = norm
        self.mel_scale = mel_scale

        assert f_min <= self.f_max, 'Require f_min: {} < f_max: {}'.format(
            f_min, self.f_max)

        fb = th.empty(0) if n_stft is None else F.create_fb_matrix(
            n_stft, self.f_min, self.f_max, self.n_mels, self.sample_rate, self.norm,
            self.mel_scale)
        self.register_buffer('fb', fb)

    def __prepare_scriptable__(self):
        if self.fb.numel() == 0:
            raise ValueError("n_stft must be provided at construction")
        return self

    def forward(self, specgram):
        shape = specgram.size()
        specgram = specgram.reshape(-1, shape[-2], shape[-1])

        if self.fb.numel() == 0:
            tmp_fb = F.create_fb_matrix(specgram.size(1), self.f_min, self.f_max,
                                        self.n_mels, self.sample_rate, self.norm,
                                        self.mel_scale)
            self.fb.resize_(tmp_fb.size())
            self.fb.copy_(tmp_fb)

        mel_specgram = th.matmul(
            specgram.transpose(1, 2),
            self.fb.type_as(specgram)
        ).transpose(1, 2)

        mel_specgram = mel_specgram.reshape(
            shape[:-2] + mel_specgram.shape[-2:])

        return mel_specgram


class MelSpectrogram(nn.Module):

    def __init__(self, n_fft, n_mels=128, sample_rate=44100, f_min=20, f_max=16000):
        super().__init__()
        self.spectrogram = Spectrogram(n_fft=n_fft)
        self.mel_scale = MelScale(
            n_mels,
            sample_rate,
            f_min,
            f_max,
            n_fft // 2 + 1,
            norm=None,
            mel_scale="htk"
        )

    def forward(self, x):
        specgram = self.spectrogram(x)
        mel_specgram = self.mel_scale(specgram)
        return mel_specgram


class MultiScaleSpectrogram(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return [Spectrogram(n_fft=2**i)(x) for i in range(6, 12)]
