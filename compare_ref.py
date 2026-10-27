"""A/B 对照实验：验证「高清中心参考图」的价值。

同一份数据、相同模型容量与迭代次数，分别训练：
  - 路线 A 模型 RefFuseSR（含 HR 中心参考图）
  - 对照模型 PlainSR（无参考图，普通单图超分）
对比它们在验证集上的 PSNR/SSIM。

运行：python compare_ref.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from config import get_config
from train import build_scenes
from data.dataset import LFSRDataset, downsample
from models.ref_fuse_sr import RefFuseSR, PlainSR
from utils.metrics import psnr, ssim


def train_one(model, opt, ds, iters, device, has_ref):
    loss_fn = torch.nn.L1Loss()
    model.train()
    for it in range(iters):
        sample = ds[it % len(ds)]
        lr, hr = sample["lr"], sample["hr"]
        B = lr.shape[0] * lr.shape[1]
        lr = lr.reshape(B, *lr.shape[2:]).to(device)
        hr = hr.reshape(B, *hr.shape[2:]).to(device)
        opt.zero_grad()
        if has_ref:
            ref = sample["ref"].unsqueeze(0).repeat(B, 1, 1, 1).to(device)
            sr = model(lr, ref)
        else:
            sr = model(lr)
        loss = loss_fn(sr, hr)
        loss.backward()
        opt.step()
    return model


def eval_model(model, lf, cfg, device, has_ref):
    s = cfg.scale
    U, V, H, W, C = lf.shape
    hp, wp = H - H % s, W - W % s
    c = U // 2
    ref_t = torch.from_numpy(lf[c, c, :hp, :wp]).permute(2, 0, 1).float().div(255).unsqueeze(0).to(device)
    ps, ss = [], []
    model.eval()
    with torch.no_grad():
        for u in range(U):
            for v in range(V):
                hr = lf[u, v, :hp, :wp]
                lr_t = torch.from_numpy(downsample(hr, s)).permute(2, 0, 1).float().div(255).unsqueeze(0).to(device)
                sr = model(lr_t, ref_t) if has_ref else model(lr_t)
                sr = torch.clamp(sr[0], 0, 1).permute(1, 2, 0).cpu().numpy()
                hr_np = hr.astype(np.float32) / 255.0
                ps.append(psnr(sr, hr_np))
                ss.append(ssim(sr, hr_np))
    return float(np.mean(ps)), float(np.mean(ss))


def main():
    cfg = get_config(dict(
        data_source="synthetic", n_scenes=8, angular=5, size=64, scale=4,
        dmax=6.0, n_feat=32, n_blocks=4, lr=1e-3, device="cpu", seed=0,
    ))
    iters = 300

    lfs = build_scenes(cfg)
    n_val = 2
    lfs_val, lfs_train = lfs[:n_val], lfs[n_val:]
    ds = LFSRDataset(lfs_train, cfg.scale, patch=None, train=True, seed=cfg.seed)

    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    model_ref = RefFuseSR(cfg.n_feat, cfg.n_blocks, cfg.scale)
    print("[train] 路线A模型（含 HR 中心参考图）...")
    train_one(model_ref, torch.optim.Adam(model_ref.parameters(), lr=cfg.lr),
              ds, iters, cfg.device, has_ref=True)

    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    model_plain = PlainSR(cfg.n_feat, cfg.n_blocks, cfg.scale)
    print("[train] 对照模型（无参考图）...")
    train_one(model_plain, torch.optim.Adam(model_plain.parameters(), lr=cfg.lr),
              ds, iters, cfg.device, has_ref=False)

    print("\n[compare] 验证集全视角平均 PSNR / SSIM:")
    rp, rs = [], []
    pp, ps2 = [], []
    for i, lf in enumerate(lfs_val):
        a, b = eval_model(model_ref, lf, cfg, cfg.device, True)
        c2, d = eval_model(model_plain, lf, cfg, cfg.device, False)
        rp.append(a); rs.append(b); pp.append(c2); ps2.append(d)
        print(f"  场景{i}: 路线A(参考图) {a:.2f} / {b:.4f}   |   无参考 {c2:.2f} / {d:.4f}")

    print(f"\n[avg] 路线A(参考图) PSNR={np.mean(rp):.2f}  SSIM={np.mean(rs):.4f}")
    print(f"[avg] 无参考        PSNR={np.mean(pp):.2f}  SSIM={np.mean(ps2):.4f}")
    print(f"[gain] 参考图带来 PSNR 提升 {np.mean(rp) - np.mean(pp):+.2f} dB")


if __name__ == "__main__":
    main()
