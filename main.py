from random import Random

from torchaudio.functional.functional import spectrogram
from karaoke.arguments import *
import torchaudio
from karaoke.dataset import KaraokeDataset
from karaoke.utils import MultiScaleSpectrogram


dataset = KaraokeDataset("./musdb", "./IR", sec=3)

x, y = dataset.__getitem__(10)
print(x.shape, y.shape)

torchaudio.save("test_mix.wav", x, sample_rate=44100)
torchaudio.save("test_music.wav", y[0:1], sample_rate=44100)
torchaudio.save("test_vocal.wav", y[2:3], sample_rate=44100)

specs = MultiScaleSpectrogram()(x.unsqueeze(0))
for spec in specs:
    print(spec.shape)
