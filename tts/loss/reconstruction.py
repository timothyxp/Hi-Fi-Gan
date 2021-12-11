from torch import Tensor, nn
from tts.collate_fn import Batch
from typing import Tuple


class WaveFormReconstructionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.waveform_loss = nn.L1Loss()

    def forward(self, batch: Batch) -> Tuple[Tensor, Tensor]:
        min_len = min(batch.waveform.shape[-1], batch.waveform_prediction.shape[-1])

        waveform_l1 = self.waveform_loss(
            batch.waveform_prediction[:, :min_len],
            batch.waveform[:, :min_len]
        )

        return waveform_l1
