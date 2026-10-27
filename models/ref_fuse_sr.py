"""路线 A 基线模型：EDSR 风格的「参考图融合」超分网络。

输入  : lr  (B, 3, h//s, w//s)   低清视角（逐视角处理，共享权重）
        ref (B, 3, h, w)         HR 中心参考图（对所有视角一致）
输出  : sr  (B, 3, h, w)         重建的高清视角

思路：把 bicubic 上采样的 LR 与参考图在通道维拼接后送入残差网络，
输出残差 + 上采样图（全局残差连接）。跨视角信息通过共享的参考图与权重流动。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.c1 = nn.Conv2d(n, n, 3, padding=1)
        self.c2 = nn.Conv2d(n, n, 3, padding=1)

    def forward(self, x):
        r = F.relu(self.c1(x), inplace=True)
        r = self.c2(r)
        return x + r


class RefFuseSR(nn.Module):
    def __init__(self, n_feat=24, n_blocks=3, scale=4):
        super().__init__()
        self.scale = scale
        self.head = nn.Conv2d(6, n_feat, 3, padding=1)   # [up_lr(3) + ref(3)]
        body = [ResBlock(n_feat) for _ in range(n_blocks)]
        body.append(nn.Conv2d(n_feat, n_feat, 3, padding=1))
        self.body = nn.Sequential(*body)
        self.tail = nn.Conv2d(n_feat, 3, 3, padding=1)

    def forward(self, lr, ref):
        up = F.interpolate(lr, size=ref.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([up, ref], dim=1)
        x = F.relu(self.head(x), inplace=True)
        x = self.body(x)
        x = self.tail(x)
        return up + x


class PlainSR(nn.Module):
    """单图超分基线（无参考图）：只输入 bicubic 上采样的 LR。

    与 RefFuseSR 结构完全相同（相同的 ResBlock 主干、相同容量），唯一区别是
    没有参考图输入（head 为 3 通道而非 6 通道）。用于对照验证参考图的价值。
    """

    def __init__(self, n_feat=24, n_blocks=3, scale=4):
        super().__init__()
        self.scale = scale
        self.head = nn.Conv2d(3, n_feat, 3, padding=1)
        body = [ResBlock(n_feat) for _ in range(n_blocks)]
        body.append(nn.Conv2d(n_feat, n_feat, 3, padding=1))
        self.body = nn.Sequential(*body)
        self.tail = nn.Conv2d(n_feat, 3, 3, padding=1)

    def forward(self, lr):
        up = F.interpolate(lr, scale_factor=self.scale, mode="bilinear", align_corners=False)
        x = F.relu(self.head(up), inplace=True)
        x = self.body(x)
        x = self.tail(x)
        return up + x
