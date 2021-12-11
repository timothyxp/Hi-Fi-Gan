from dataclasses import dataclass


@dataclass
class Config:
    batch_size: int
    num_workers: int = 4

    n_mels: int = 80
    n_channels: int = 512

    leaky: float = 0.1

    upsample_size = [8, 8, 2, 2]
    kernel_u = [16, 16, 4, 4]
    kernel_r = [3, 7, 11]
    dilations_r = [[1, 1], [3, 1], [5, 1]] * 3

