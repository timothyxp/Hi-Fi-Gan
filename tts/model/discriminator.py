from torch import nn
from tts.config import Config
from .discriminator_layers import MCD, MPD


class Discriminator(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.mcd = MCD(config)
        self.mpd = MPD(config)
