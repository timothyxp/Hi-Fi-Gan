from torch import Tensor, nn
from tts.collate_fn import Batch
from typing import Tuple
from torch.nn import functional as F


class WaveFormReconstructionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.waveform_loss = nn.L1Loss()

    def forward(self, batch: Batch) -> Tensor:
        diff_len = batch.waveform_prediction.shape[-1] - batch.waveform.shape[-1]
        waveform = F.pad(batch.waveform, (0, diff_len))

        waveform_l1 = self.waveform_loss(
            batch.waveform_prediction,
            waveform
        )

        return waveform_l1
