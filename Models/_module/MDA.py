import math
import torch
from torch import nn
from torch.nn import functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def apply_attention_C(x, scores, N):
    B, C, T, H, W = x.shape
    x = x.view(B, C // N, N, T, H, W)
    scores = scores.view(B, C // N, N, 1, 1, 1)
    scores = F.softmax(scores, dim=2)
    return (x * scores).sum(dim=2)

def apply_attention_T(x, scores, N):
    B, C, T, H, W = x.shape
    x = x.view(B, C, T // N, N, H, W)
    scores = scores.view(B, 1, T // N, N, 1, 1)
    scores = F.softmax(scores, dim=3)
    return (x * scores).sum(dim=3)

def apply_attention_HW(x, scores, N):
    B, C, T, H, W = x.shape
    x = x.view(B, C, T, H // N, N, W // N, N).permute(0,1,2,3,5,4,6)
    x = x.reshape(B, C, T, (H//N)*(W//N), N*N)
    scores = scores.view(B, 1, 1, H//N, N, W//N, N).permute(0,1,2,3,5,4,6)
    scores = scores.reshape(B, 1, 1, (H//N)*(W//N), N*N)
    scores = F.softmax(scores, dim=-1)
    x = torch.einsum('bctpq,bctpq->bctp', x, scores)
    return x.view(B, C, T, H // N, W // N)

def restore_HW(x, N):
    B, C, T, H, W = x.shape
    x = x.unsqueeze(-1).unsqueeze(-1).expand(B, C, T, H, W, N, N)
    return x.permute(0,1,2,3,5,4,6).reshape(B, C, T, H*N, W*N)

def restore_C(x, N):
    return x.repeat_interleave(N, dim=1)

def restore_T(x, N):
    return x.repeat_interleave(N, dim=2)

class _MDA(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.query_conv = nn.Conv3d(in_dim, in_dim // 2, kernel_size=1)
        self.key_conv = nn.Conv3d(in_dim, in_dim // 2, kernel_size=1)
        self.value_conv = nn.Conv3d(in_dim, in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.tensor(0.1))

    def forward(self, x, cs, fs, ss):
        out = apply_attention_C(x, cs, 2)
        out = apply_attention_T(out, fs, 2)
        out = apply_attention_HW(out, ss, 2)

        B, C, T, H, W = out.shape
        proj_q = self.query_conv(out).view(B, -1, T*H*W).permute(0, 2, 1)
        proj_k = self.key_conv(out).view(B, -1, T*H*W)
        scale = 1 / math.sqrt(C // 8)
        attention = F.softmax(torch.bmm(proj_q, proj_k) * scale, dim=-1)
        proj_v = self.value_conv(out).view(B, -1, T*H*W).permute(0, 2, 1)
        out = torch.bmm(attention, proj_v).view(B, C, T, H, W)
        out = self.gamma * out

        out = restore_HW(out, 2)
        out = restore_T(out, 2)
        out = restore_C(out, 2)
        return out
