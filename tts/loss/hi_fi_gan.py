from torch import nn
from tts.collate_fn import Batch
from tts.model.discriminator_layers import MCD, MPD
from torch.nn import functional as F
from .reconstruction import WaveFormReconstructionLoss


class HiFiGANLoss(nn.Module):
    def __init__(self, config, reconstruction_loss):
        super().__init__()
        self.waveform_loss = reconstruction_loss
        self.fm_loss = nn.L1Loss()
        self.lambda_mel = config.lambda_mel
        self.lambda_fm = config.lambda_fm

    @staticmethod
    def calc_gan_loss(mcd_losses, mpd_losses, true=True):
        if true:
            return sum(((mcd_loss - 1) ** 2).mean() for mcd_loss in mcd_losses) + \
                sum(((mpd_loss - 1) ** 2).mean() for mpd_loss in mpd_losses)
        else:
            return sum((mcd_loss ** 2).mean() for mcd_loss in mcd_losses) + \
                sum((mpd_loss ** 2).mean() for mpd_loss in mpd_losses)
        
    def feature_matching_loss(self, fake_fms_arr, true_fms_arr):
        return sum(sum([self.fm_loss(fake_fm, true_fm) for fake_fm, true_fm in zip(fake_fms, true_fms)]) for fake_fms, true_fms in zip(fake_fms_arr, true_fms_arr))

    def forward(self, batch: Batch, mpd: MPD, mcd: MCD, generator_step: bool = False):
        if generator_step:
            waveform_l1 = self.waveform_loss(batch)

            mpd_fake, mpd_fake_feature_map = mpd(batch.waveform_prediction.unsqueeze(1))
            mcd_fake, mcd_fake_feature_map = mcd(batch.waveform_prediction.unsqueeze(1))
            
            diff_len = batch.waveform_prediction.shape[-1] - batch.waveform.shape[-1]
            waveform = F.pad(batch.waveform, (0, diff_len))

            mpd_true, mpd_true_feature_map = mpd(waveform.unsqueeze(1))
            mcd_true, mcd_true_feature_map = mcd(waveform.unsqueeze(1))
            
            fm_loss = self.feature_matching_loss(mpd_fake_feature_map, mpd_true_feature_map) \
                + self.feature_matching_loss(mcd_fake_feature_map, mcd_true_feature_map)

            gan_loss = self.calc_gan_loss(mcd_fake, mpd_fake, true=True)

            return gan_loss, fm_loss * self.lambda_fm, waveform_l1 * self.lambda_mel

        else:
            mpd_fake, mpd_fake_feature_map = mpd(batch.waveform_prediction.unsqueeze(1).detach())
            mcd_fake, mcd_fake_feature_map = mcd(batch.waveform_prediction.unsqueeze(1).detach())

            mpd_true, mpd_true_feature_map = mpd(batch.waveform.unsqueeze(1))
            mcd_true, mcd_true_feature_map = mcd(batch.waveform.unsqueeze(1))

            gan_true_loss = self.calc_gan_loss(mcd_true, mpd_true, true=True)
            gan_false_loss = self.calc_gan_loss(mcd_fake, mpd_fake, true=False)

            return gan_true_loss, gan_false_loss
