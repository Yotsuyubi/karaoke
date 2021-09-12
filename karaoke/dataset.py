from random import Random
from karaoke.arguments import *
import musdb
import torch as th
import torchaudio
from torch.utils.data import Dataset


class KaraokeDataset(Dataset):

    def __init__(self, musdb_root, ir_root, subset="train", sec=10, sample_rate=44100):
        self.mus = musdb.DB(download=True, root=musdb_root, subsets=subset)
        self.sample_length = sample_rate*sec
        self.IR = IR(ir_root)

    def __len__(self):
        return len(self.mus)

    def __getitem__(self, index):

        track = self.mus[index]

        music = th.zeros((2, self.sample_length)).float()
        vocal = th.zeros((2, self.sample_length)).float()
        mix = th.zeros((2, self.sample_length)).float()

        for stem in range(1, 5):

            params = self.generate_random_params()

            sample = track

            if params["remix"]:
                sample = self.mus[params["track_index"]]

            x = th.tensor(sample.stems[stem]).transpose(-1, -2).float()
            x = RandomCrop(self.sample_length)(x)

            post_process = None
            if params["IR_post_process"]:
                post_process = EQ(*params["IR_eq_params"])
            if params["IR"]:
                x = self.IR(
                    post_process=post_process,
                    wet_rate=params["IR_wet_rate"]
                )(x)

            if params["random_eq"]:
                x = EQ(*params["eq_params"])(x)
            if params["channel_swap"]:
                x = ChannelSwap()(x)
            if params["sign_swap"]:
                x = SignSwap()(x)
            if params["random_shift"]:
                x = Shifter(*params["shifter_params"])(x)
            if params["random_volume"]:
                x = Fader(*params["volume_params"])(x)
            if params["random_time_stretch"]:
                x = TimeStretch(*params["time_stretch_params"])(x)
            if params["random_pitch_shift"]:
                x = PitchShift(*params["pitch_shift_params"])(x)
            if params["mute"]:
                x = th.zeros((2, self.sample_length)).float()

            if stem != 4:
                music[:, :] += x
            else:
                vocal[:, :] += x

        mix += (music + vocal)

        return mix, th.cat([music, vocal], dim=0)

    def generate_random_params(self):
        return {
            "remix": random.random() < 0.2,
            "track_index": random.randint(0, self.__len__()),
            "random_eq": random.random() < 0.2,
            "eq_params": ((random.random()*2.0-1.0)*24.0, random.randint(20, 16000)),
            "channel_swap": random.random() < 0.2,
            "sign_swap": random.random() < 0.2,
            "random_shift": random.random() < 0.2,
            "shifter_params": (random.random()*2.0-1.0,),
            "random_volume": random.random() < 0.2,
            "volume_params": (random.randint(250, 1250)/1000,),
            "random_time_stretch": random.random() < 0.2,
            "time_stretch_params": (random.randint(880, 1120)/1000,),
            "random_pitch_shift": random.random() < 0.2,
            "pitch_shift_params": (random.randint(-2, 2),),
            "mute": random.random() < 0.2,
            "IR_post_process": random.random() < 0.2,
            "IR": random.random() < 0.2,
            "IR_eq_params": ((random.random()*2.0-1.0)*24.0, random.randint(20, 16000)),
            "IR_wet_rate": random.random()*0.5
        }
