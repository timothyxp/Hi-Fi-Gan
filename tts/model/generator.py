from torch import nn
from tts.config import Config
from tts.collate_fn.collate import Batch
from torch.nn import functional as F
from .generator_layers import MRF


class Generator(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        self.pre_net = nn.Conv1d(config.n_mels, config.n_channels, kernel_size=7, stride=1, padding=3)

        self.up_convs = nn.ModuleList()
        self.mrfs = nn.ModuleList()
        self.leaky = config.leaky

        n_channels = config.n_channels

        for i in range(len(config.kernel_u)):
            out_channels = n_channels // 2
            self.up_convs.append(nn.ConvTranspose1d(
                n_channels,
                out_channels,
                kernel_size=config.kernel_u[i],
                stride=config.upsample_size[i]
            ))

            self.mrfs.append(MRF(out_channels, config.kernel_r, config.dilations_r, config.leaky))
            
            n_channels = out_channels

        self.post_net = nn.Conv1d(n_channels, 1, kernel_size=7, stride=1, padding=3)

    def forward(self, batch: Batch):
        x = self.pre_net(batch.melspec)

        for up_conv, mrf in zip(self.up_convs, self.mrfs):
            F.leaky_relu_(x, self.leaky)

            print("before upconv", x.shape)
            x = up_conv(x)
            print("after upconv", x.shape)

            x = mrf(x)

        F.leaky_relu_(x, self.leaky)
        x = self.post_net(x).unsqueeze(1)
        batch.waveform_prediction = F.tanh(x)

        return batch
