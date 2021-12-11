from dataclasses import dataclass
from typing import Tuple, Dict, Optional, List, Union

import torch
from torch.nn.utils.rnn import pad_sequence


@dataclass
class Batch:
    waveform: torch.Tensor
    waveform_length: torch.Tensor

    melspec: torch.Tensor = None
    waveform_prediction: Optional[torch.Tensor] = None

    def to(self, device: torch.device, non_blocking=True) -> 'Batch':
        self.waveform = self.waveform.to(device, non_blocking=non_blocking)
        self.waveform_length = self.waveform_length.to(device, non_blocking=non_blocking)

        if self.melspec is not None:
            self.melspec = self.melspec.to(device, non_blocking=non_blocking)

        if self.waveform_prediction is not None:
            self.waveform_prediction = self.waveform_prediction.to(device, non_blocking=non_blocking)

        return self


class LJSpeechCollator:
    def __call__(self, instances: List[Tuple]) -> Batch:
        waveform, waveform_length = list(
            zip(*instances)
        )

        waveform = pad_sequence([
            waveform_[0] for waveform_ in waveform
        ]).transpose(0, 1)
        waveform_length = torch.cat(waveform_length)

        return Batch(waveform, waveform_length)


class TestCollator:
    def __call__(self, instances: List[Tuple]) -> Batch:
        tokens, token_lengths, transcript = list(zip(*instances))

        empty = torch.zeros(0)
        
        tokens = pad_sequence([
            tokens_[0] for tokens_ in tokens
        ]).transpose(0, 1)
        token_lengths = torch.cat(token_lengths)

        return Batch(empty, empty, transcript, tokens, token_lengths)
