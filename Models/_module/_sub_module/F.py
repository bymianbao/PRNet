import torch
import torch.nn as nn
import torch.nn.functional as F

class Fattention(nn.Module):
    def __init__(self, in_temp, k=4):
        super().__init__()
        self.k = k
        self.fc = nn.Sequential(
            nn.Linear(in_temp, in_temp // 2),
            nn.ReLU(inplace=True),
            nn.Linear(in_temp // 2, in_temp),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, f, h, w = x.size()

        x_swapped = x.permute(0, 2, 1, 3, 4)
        x_flatten = x_swapped.reshape(b, f, -1)
        topk_values, _ = torch.topk(x_flatten, k=self.k, dim=2)
        topk_mean = topk_values.sum(dim=-1)
        topk_mean = topk_mean.view(b, f)

        avg_pool = x.mean(dim=1, keepdim=True).mean(dim=3, keepdim=True).mean(dim=4, keepdim=True)
        avg_out = self.fc(avg_pool.view(b, -1))

        out = self.fc(topk_mean.view(b, -1)) + avg_out
        out = self.sigmoid(out).view(b, 1, out.size(1), 1, 1)

        return x * out, out
