import torchaudio
import torch
import random


class LJSpeechDataset(torchaudio.datasets.LJSPEECH):
    def __init__(self, root, max_len: int = 8192 * 2):
        super().__init__(root=root)
        self.max_len = max_len

    def __getitem__(self, index: int):
        waveform, _, _, _ = super().__getitem__(index)
        waveform_length = torch.tensor([waveform.shape[-1]]).int()
        
        if self.max_len is not None:
            start_idx = random.randint(0, waveform.shape[1] - self.max_len)
            waveform = waveform[:, start_idx: start_idx + self.max_len]

        return waveform, waveform_length
