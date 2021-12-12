from torch import nn
from tts.collate_fn import Batch
from tts.model.discriminator_layers import MCD, MPD
from .reconstruction import WaveFormReconstructionLoss


class HiFiGANLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.waveform_loss = WaveFormReconstructionLoss()
        self.feature_matching_loss = nn.L1Loss()
        self.lambda_mel = config.lambda_mel
        self.lambda_fm = config.lambda_fm

    @staticmethod
    def calc_gan_loss(mcd_loss, mpd_loss, true=True):
        if true:
            return (mcd_loss - 1) ** 2 + (mpd_loss - 1) ** 2
        else:
            return mcd_loss ** 2 + mpd_loss ** 2

    def forward(self, batch: Batch, mpd: MPD, mcd: MCD, generator_step: bool = False):
        if generator_step:
            waveform_l1 = self.waveform_loss(batch)

            mpd_fake, mpd_fake_feature_map = mpd(batch.waveform_prediction)
            mcd_fake, mcd_fake_feature_map = mcd(batch.waveform_prediction)

            mpd_true, mpd_true_feature_map = mpd(batch.waveform)
            mcd_true, mcd_true_feature_map = mcd(batch.waveform)

            fm_loss = self.feature_matching_loss(mpd_fake_feature_map, mpd_true_feature_map) \
                + self.feature_matching_loss(mcd_fake_feature_map, mcd_true_feature_map)

            gan_loss = self.calc_gan_loss(mcd_fake, mpd_fake, true=True)

            return gan_loss, fm_loss * self.lambda_fm, waveform_l1 * self.lambda_mel

        else:
            mpd_fake, mpd_fake_feature_map = mpd(batch.waveform_prediction.detach())
            mcd_fake, mcd_fake_feature_map = mcd(batch.waveform_prediction.detach())

            mpd_true, mpd_true_feature_map = mpd(batch.waveform)
            mcd_true, mcd_true_feature_map = mcd(batch.waveform)

            gan_true_loss = self.calc_gan_loss(mcd_true, mpd_true, true=True)
            gan_false_loss = self.calc_gan_loss(mcd_fake, mpd_fake, true=False)

            return gan_true_loss, gan_false_loss
