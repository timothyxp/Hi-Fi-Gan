from torch import nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm


def calc_padding(kernel_size, dilation):
    return dilation * (kernel_size - 1) // 2


class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilations, leaky: float):
        super().__init__()
        self.leaky = leaky

        self.convs1 = nn.ModuleList([
            weight_norm(nn.Conv1d(
                channels, channels, kernel_size, stride=1,
                dilation=dilation[0], padding=calc_padding(kernel_size, dilation[0])
            ))
            for dilation in dilations
        ])

        self.convs2 = nn.ModuleList([
            weight_norm(nn.Conv1d(
                channels, channels, kernel_size, stride=1,
                dilation=dilation[1], padding=calc_padding(kernel_size, dilation[1])
            ))
            for dilation in dilations
        ])

    def forward(self, x):
        for conv1, conv2 in zip(self.convs1, self.convs2):
            conv_x = F.leaky_relu(x, self.leaky)
            conv_x = conv1(conv_x)
            conv_x = F.leaky_relu(conv_x, self.leaky)
            conv_x = conv2(conv_x)

            x = x + conv_x

        return x


class MRF(nn.Module):
    def __init__(self, channels, kernels, dilations, leaky):
        super().__init__()
        self.resblocks = nn.ModuleList([
            ResBlock(channels, kernel_size, dilation, leaky) for kernel_size, dilation in zip(kernels, dilations)
        ])

    def forward(self, x):
        res = self.resblocks[0](x)

        for i in range(1, len(self.resblocks)):
            res += self.resblocks[i](x)

        return res
