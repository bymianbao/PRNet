import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, k=4):
        super().__init__()
        self.k = k
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 2, in_channels),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, d, h, w = x.size()
        avg_out = self.avg_pool(x).view(b, c)
        avg_score = self.fc(avg_out)

        x_flatten = x.view(b, c, -1)
        topk_values, _ = torch.topk(x_flatten, k=self.k, dim=-1)
        topk_mean = topk_values.sum(dim=-1)
        topk_score = self.fc(topk_mean)

        out = avg_score + topk_score
        out = self.sigmoid(out).view(b, c, 1, 1, 1)
        return x * out, out
