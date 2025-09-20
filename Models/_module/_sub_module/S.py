import torch
import torch.nn as nn

class CDReducer(nn.Module):
    def __init__(self, k=4):
        super().__init__()
        self.k = k

    def forward(self, x):
        b, c, d, h, w = x.size()
        x_flatten = x.view(b, c * d, h, w)
        topk_values, _ = torch.topk(x_flatten, k=self.k, dim=1, largest=True)
        topk_mean = topk_values.sum(dim=1, keepdim=True).unsqueeze(dim=1)
        mean_CD = torch.mean(x, dim=(1, 2), keepdim=True)
        return topk_mean, mean_CD

class Sattention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.CD_precess = CDReducer()
        self.fc = nn.Sequential(
            nn.Conv3d(2, 1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(1, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        max_s, mean_s = self.CD_precess(x)
        attention_map = torch.cat([max_s, mean_s], dim=1)
        attention_map = self.fc(attention_map)
        attention_map_view = attention_map.expand_as(x)
        return x * attention_map_view, attention_map
