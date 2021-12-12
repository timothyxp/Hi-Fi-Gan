from dataclasses import dataclass


@dataclass
class Config:
    batch_size: int
    device: str
    num_workers: int = 4

    n_mels: int = 80
    n_channels: int = 512

    leaky: float = 0.1

    upsample_size = [8, 8, 2, 2]
    kernel_u = [16, 16, 4, 4]
    kernel_r = [3, 7, 11]
    dilations_r = [[[1, 1], [3, 1], [5, 1]]] * 3

    mpd_kernel_size: int = 5
    mpd_stride: int = 3
    mpd_periods = [2, 3, 5, 7, 11]

    mcd_num_layers = 3

    lambda_fm: float = 2.
    lambda_mel: float = 45
    
    wandb_project = 'Hi-Fi-GAN'
    
    n_epochs: int = 100
    lr = 4e-3
    
    overfit_batch = True

    def get(self, attr, default_value=None):
        return getattr(self, attr, default_value)
    
    def __getitem__(self, key):
        if hasattr(self, key):
            return getattr(self, key)
        
        raise KeyError(key)
