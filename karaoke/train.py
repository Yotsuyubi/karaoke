from pytorch_lightning.trainer import Trainer
import pytorch_lightning as pl
from .model import WaveUNet, PostNet, WaveDiscriminator, SpectrogramDiscriminator
from .utils import MultiScaleSpectrogram
from torch.nn import functional as F
import torch as th
from .dataset import KaraokeDataset
from torch.utils.data import DataLoader


class LitGAN(pl.LightningModule):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.wave_u_net = WaveUNet()
        self.post_net = PostNet()
        self.wave_discriminator = WaveDiscriminator()
        self.spectrogram_discriminator = SpectrogramDiscriminator()

        self.multiscale_spectrogram = MultiScaleSpectrogram()

    def forward(self, x):
        x = self.wave_u_net(x)
        return self.post_net(x), x

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def construction_loss(self, y_hat, y):
        return F.l1_loss(y_hat, y)

    def spectrogram_loss(self, y_hat, y):
        y_hats = self.multiscale_spectrogram(y_hat)
        ys = self.multiscale_spectrogram(y)
        return th.tensor([F.l1_loss(y_hat, y) for y_hat, y in zip(y_hats, ys)]).type_as(y_hat).sum()

    def training_step(self, batch, batch_idx, optimizer_idx):
        x, y = batch

        # train generator
        if optimizer_idx == 0:

            valid = th.ones(x.size(0), 1)
            valid = valid.type_as(x)

            y_hat_postnet, y_hat = self(x)

            l1_loss = self.construction_loss(y_hat, y)
            postnet_l1_loss = self.construction_loss(y_hat_postnet, y)
            spectrogram_loss = self.spectrogram_loss(y_hat, y)
            postnet_spectrogram_loss = self.spectrogram_loss(y_hat_postnet, y)
            wave_adversarial_loss = self.adversarial_loss(
                self.wave_discriminator(y_hat_postnet),
                valid
            )
            spectrogram_adversarial_loss = th.tensor([
                self.adversarial_loss(
                    self.wave_discriminator(y_hat_spectrogram),
                    valid
                ) for y_hat_spectrogram in self.multiscale_spectrogram(y_hat_postnet)
            ]).type_as(x).sum()

            loss = l1_loss + postnet_l1_loss + spectrogram_loss + \
                postnet_spectrogram_loss + wave_adversarial_loss + spectrogram_adversarial_loss

            self.log('l1_loss', postnet_l1_loss, prog_bar=True)

            return loss

        # train critic
        if optimizer_idx > 0:

            valid = th.ones(x.size(0), 1)
            valid = valid.type_as(x)

            real_wave_loss = self.adversarial_loss(
                self.wave_discriminator(y),
                valid
            )
            real_spectrogram_loss = th.tensor([
                self.adversarial_loss(
                    self.wave_discriminator(y_spectrogram),
                    valid
                ) for y_spectrogram in self.multiscale_spectrogram(y)
            ]).type_as(x).sum()

            fake = th.zeros(x.size(0), 1)
            fake = fake.type_as(x)
            y_hat = self(x)[0].detach()

            fake_wave_loss = self.adversarial_loss(
                self.wave_discriminator(y_hat),
                valid
            )
            fake_spectrogram_loss = th.tensor([
                self.adversarial_loss(
                    self.wave_discriminator(y_spectrogram),
                    valid
                ) for y_spectrogram in self.multiscale_spectrogram(y_hat)
            ]).type_as(x).sum()

            loss = (real_wave_loss + real_spectrogram_loss +
                    fake_wave_loss + fake_spectrogram_loss) / 2

            self.log('critic_loss', loss, prog_bar=True)

            return loss

    def configure_optimizers(self):
        opt_g = th.optim.Adam(
            list(self.wave_u_net.parameters())+list(self.post_net.parameters()), lr=1e-4
        )
        opt_d = th.optim.Adam(
            list(self.wave_discriminator.parameters())+list(self.spectrogram_discriminator.parameters()), lr=1e-3
        )
        return [opt_g, opt_d], []


if __name__ == "__main__":

    dataset = KaraokeDataset("./musdb", "./IR", sec=10)

    train_loader = DataLoader(dataset, batch_size=32,
                              shuffle=True, num_workers=1)

    gan = LitGAN()

    # training
    trainer = pl.Trainer(
        gpus=0,
    )

    trainer.fit(gan, train_loader)
    trainer.save_checkpoint("gan.ckpt")
