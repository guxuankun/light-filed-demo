"""构造路线 A 训练对：LR 全部视角 + HR 中心参考 -> HR 全部视角。

降质模型：对每个视角用 bicubic 下采样 s 倍得到 LR；中心视角额外保留 HR 原图作参考。
所有视角都有 HR 真值，可正常监督训练。
"""
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset


def downsample(img, scale):
    """img: (H,W,C) uint8 -> (H//s, W//s, C) uint8，bicubic 下采样。"""
    h, w = img.shape[:2]
    return cv2.resize(img, (w // scale, h // scale), interpolation=cv2.INTER_CUBIC)


class LFSRDataset(Dataset):
    """一个样本 = 一个光场。

    返回 dict:
      lr  : (U,V,C, h//s, w//s)  全部视角的低清版本
      ref : (C, h, w)            HR 中心视角（参考图）
      hr  : (U,V,C, h, w)        全部视角的高清真值
    """

    def __init__(self, light_fields, scale=4, patch=None, train=True, seed=0):
        self.lfs = light_fields
        self.scale = scale
        self.patch = patch
        self.train = train
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.lfs)

    def __getitem__(self, idx):
        lf = self.lfs[idx]                      # (U,V,H,W,C)
        U, V, H, W, C = lf.shape
        s = self.scale

        if self.train and self.patch:
            y = self.rng.integers(0, H - self.patch + 1)
            x = self.rng.integers(0, W - self.patch + 1)
            hr = lf[:, :, y:y + self.patch, x:x + self.patch, :]
        else:
            hp, wp = H - H % s, W - W % s       # 裁到能被 s 整除
            hr = lf[:, :, :hp, :wp, :]

        U2, V2, hp, wp, C = hr.shape
        lr = np.zeros((U2, V2, hp // s, wp // s, C), dtype=np.uint8)
        for u in range(U2):
            for v in range(V2):
                lr[u, v] = downsample(hr[u, v], s)

        ref = hr[U2 // 2, V2 // 2]              # HR 中心视角

        lr = torch.from_numpy(lr).permute(0, 1, 4, 2, 3).float().div(255.0)
        hr = torch.from_numpy(hr).permute(0, 1, 4, 2, 3).float().div(255.0)
        ref = torch.from_numpy(ref).permute(2, 0, 1).float().div(255.0)
        return {"lr": lr, "ref": ref, "hr": hr}
