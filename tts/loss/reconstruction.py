from torch import Tensor, nn
from tts.collate_fn import Batch
from torch.nn import functional as F


class WaveFormReconstructionLoss(nn.Module):
    def __init__(self, featurizer):
        super().__init__()
        self.waveform_loss = nn.L1Loss()
        self.featurizer = featurizer

    def forward(self, batch: Batch) -> Tensor:
        melspec_prediction = self.featurizer(batch.waveform_prediction)

        # diff_len = batch.waveform_prediction.shape[-1] - batch.waveform.shape[-1]
        # waveform = F.pad(batch.waveform, (0, diff_len))

        waveform_l1 = self.waveform_loss(
            batch.melspec,
            melspec_prediction
        )

        return waveform_l1
