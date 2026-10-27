"""评估与可视化：在单个光场上计算 ours vs bicubic 的 PSNR/SSIM，并保存对比图。"""
import os
import numpy as np
import cv2
import torch

from utils.metrics import psnr, ssim
from data.dataset import downsample


def _up_bicubic(lr_np, scale):
    h, w = lr_np.shape[:2]
    return cv2.resize(lr_np, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)


def evaluate(model, lf, cfg, device="cpu"):
    """在单个光场 lf (U,V,H,W,3 uint8) 上评估全视角平均指标。

    返回 dict: ours_psnr/ssim, bic_psnr/ssim（全视角平均）。
    """
    model.eval()
    s = cfg.scale
    U, V, H, W, C = lf.shape
    hp, wp = H - H % s, W - W % s
    c = U // 2
    ref = lf[c, c, :hp, :wp]
    ref_t = torch.from_numpy(ref).permute(2, 0, 1).float().div(255).unsqueeze(0).to(device)

    ours_p, ours_s, bic_p, bic_s = [], [], [], []
    with torch.no_grad():
        for u in range(U):
            for v in range(V):
                hr = lf[u, v, :hp, :wp]
                lr = downsample(hr, s)
                lr_t = torch.from_numpy(lr).permute(2, 0, 1).float().div(255).unsqueeze(0).to(device)
                sr = torch.clamp(model(lr_t, ref_t)[0], 0, 1).permute(1, 2, 0).cpu().numpy()

                hr_np = hr.astype(np.float32) / 255.0
                bic = _up_bicubic(lr, s).astype(np.float32) / 255.0
                ours_p.append(psnr(sr, hr_np))
                ours_s.append(ssim(sr, hr_np))
                bic_p.append(psnr(bic, hr_np))
                bic_s.append(ssim(bic, hr_np))

    return dict(
        ours_psnr=float(np.mean(ours_p)), ours_ssim=float(np.mean(ours_s)),
        bic_psnr=float(np.mean(bic_p)), bic_ssim=float(np.mean(bic_s)),
    )


def save_comparison(model, lf, cfg, out_path, device="cpu", views=None):
    """保存指定视角的 bicubic / ours / GT 三列对比图。"""
    s = cfg.scale
    U, V, H, W, C = lf.shape
    hp, wp = H - H % s, W - W % s
    c = U // 2
    if views is None:
        views = [(c, U - 1), (0, 0)]          # 默认：中心视角 + 一个对角边缘视角
    ref = lf[c, c, :hp, :wp]
    ref_t = torch.from_numpy(ref).permute(2, 0, 1).float().div(255).unsqueeze(0).to(device)

    model.eval()
    rows = []
    with torch.no_grad():
        for (u, v) in views:
            hr = lf[u, v, :hp, :wp]
            lr = downsample(hr, s)
            lr_t = torch.from_numpy(lr).permute(2, 0, 1).float().div(255).unsqueeze(0).to(device)
            sr = torch.clamp(model(lr_t, ref_t)[0], 0, 1).permute(1, 2, 0).cpu().numpy()
            bic = _up_bicubic(lr, s).astype(np.float32) / 255.0
            hr_np = hr.astype(np.float32) / 255.0
            rows.append(np.concatenate([bic, sr, hr_np], axis=1))

    grid = np.concatenate(rows, axis=0)
    img = (grid * 255).clip(0, 255).astype(np.uint8)

    # 顶部加一列标签条
    bar_h = 26
    label_img = np.full((bar_h, img.shape[1], 3), 245, dtype=np.uint8)
    col_w = hp
    for i, name in enumerate(["bicubic", "ours (ref-SR)", "GT"]):
        x0 = i * col_w + 8
        cv2.putText(label_img, name, (x0, 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (60, 60, 60), 1, cv2.LINE_AA)
    img = np.concatenate([label_img, img], axis=0)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, img)
    return out_path
