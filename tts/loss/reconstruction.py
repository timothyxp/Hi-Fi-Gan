from torch import Tensor, nn
from tts.collate_fn import Batch
from torch.nn import functional as F


class WaveFormReconstructionLoss(nn.Module):
    def __init__(self, featurizer, pad_value):
        super().__init__()
        self.waveform_loss = nn.L1Loss()
        self.featurizer = featurizer
        self.pad_value = pad_value

    def forward(self, batch: Batch) -> Tensor:
        melspec_prediction = self.featurizer(batch.waveform_prediction)
        melspec = batch.melspec

        diff_len = melspec_prediction.shape[-1] - batch.melspec.shape[-1]
        if diff_len > 0:
            melspec_prediction = F.pad(melspec_prediction, (0, diff_len), self.pad_value)
        else:
            melspec = F.pad(melspec, (0, -diff_len), self.pad_value)

        waveform_l1 = self.waveform_loss(
            melspec,
            melspec_prediction
        )

        return waveform_l1
