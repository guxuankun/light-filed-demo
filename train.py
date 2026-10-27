"""训练脚本（路线 A：LR 全部视角 + HR 中心参考 -> HR 全部视角）。

用法示例：
  # 合成光场快速训练（CPU）
  python train.py --data_source synthetic --iters 200

  # HCI 真实数据训练（先把 .h5 放到 data/hci/ 下）
  python train.py --data_source hci --angular 9 --size 512 --patch 64 --iters 200000 --device cuda
"""
import os
import sys
import time
import argparse



sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn

from config import get_config, add_common_args
from data.synthetic import make_light_field
from data.hci import load_hci_dataset
from data.dataset import LFSRDataset
from models.ref_fuse_sr import RefFuseSR
from eval import evaluate, save_comparison


def build_scenes(cfg):
    if cfg.data_source == "synthetic":
        return [make_light_field(cfg.angular, cfg.size, cfg.dmax, seed=cfg.seed + i)[0]
                for i in range(cfg.n_scenes)]
    return load_hci_dataset(cfg.data_dir)


def train_model(cfg, lfs_train, lfs_val):
    device = cfg.device
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    model = RefFuseSR(cfg.n_feat, cfg.n_blocks, cfg.scale).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.L1Loss()
    train_ds = LFSRDataset(lfs_train, cfg.scale, patch=cfg.patch, train=True, seed=cfg.seed)

    model.train()
    t0 = time.time()
    log_every = max(1, cfg.iters // 10)
    for it in range(cfg.iters):
        sample = train_ds[it % len(train_ds)]
        lr = sample["lr"]
        hr = sample["hr"]
        B = lr.shape[0] * lr.shape[1]                     # U*V 个视角
        lr = lr.reshape(B, *lr.shape[2:]).to(device)
        hr = hr.reshape(B, *hr.shape[2:]).to(device)
        ref = sample["ref"].unsqueeze(0).repeat(B, 1, 1, 1).to(device)

        opt.zero_grad()
        sr = model(lr, ref)
        loss = loss_fn(sr, hr)
        loss.backward()
        opt.step()

        if it == 0 or (it + 1) % log_every == 0:
            print(f"  iter {it + 1}/{cfg.iters}  loss={loss.item():.5f}  "
                  f"elapsed={time.time() - t0:.1f}s", flush=True)
    return model


def main():
    p = argparse.ArgumentParser()
    add_common_args(p)
    cfg = get_config(vars(p.parse_args()))
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    os.makedirs(cfg.out_dir, exist_ok=True)

    lfs = build_scenes(cfg)
    n_val = max(1, int(len(lfs) * cfg.val_split))
    lfs_val, lfs_train = lfs[:n_val], lfs[n_val:]
    print(f"[data] {cfg.data_source}: train={len(lfs_train)} val={len(lfs_val)}  "
          f"angular={cfg.angular} size={cfg.size} scale={cfg.scale} device={cfg.device}")

    model = train_model(cfg, lfs_train, lfs_val)

    print("[eval] 验证集全视角平均指标:")
    op, os_, bp, bs = [], [], [], []
    for lf in lfs_val:
        r = evaluate(model, lf, cfg, cfg.device)
        op.append(r["ours_psnr"]); os_.append(r["ours_ssim"])
        bp.append(r["bic_psnr"]); bs.append(r["bic_ssim"])
    print(f"  bicubic : PSNR={np.mean(bp):.2f}  SSIM={np.mean(bs):.4f}")
    print(f"  ours    : PSNR={np.mean(op):.2f}  SSIM={np.mean(os_):.4f}")

    ckpt = os.path.join(cfg.ckpt_dir, f"reffuse_sr_x{cfg.scale}.pt")
    torch.save({"cfg": vars(cfg), "state_dict": model.state_dict()}, ckpt)
    print(f"[ckpt] saved -> {ckpt}")

    out_png = os.path.join(cfg.out_dir, f"compare_x{cfg.scale}.png")
    save_comparison(model, lfs_val[0], cfg, out_png, cfg.device)
    print(f"[img] saved -> {out_png}")


if __name__ == "__main__":
    main()
