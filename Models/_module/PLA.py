import torch
from torch import nn
from Models._module._sub_module.C import ChannelAttention
from Models._module._sub_module.F import Fattention
from Models._module._sub_module.S import Sattention

class _PLA(nn.Module):
    def __init__(self, inchannels, intemp):
        super().__init__()
        self.Cattention = ChannelAttention(in_channels=inchannels)
        self.Fattention = Fattention(in_temp=intemp)
        self.Sattention = Sattention(in_channels=inchannels)

    def forward(self, x):
        x1, cs = self.Cattention(x)
        x2, fs = self.Fattention(x1)
        x3, ss = self.Sattention(x2)
        return x3, cs, fs, ss
