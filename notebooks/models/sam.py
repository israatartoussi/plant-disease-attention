
import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        attention_map = torch.sigmoid(self.conv(x_cat))
        return attention_map * x

class SAM(nn.Module):
    def __init__(self, in_planes):
        super(SAM, self).__init__()
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        return self.spatial_attention(x)
