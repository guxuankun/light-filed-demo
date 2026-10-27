"""评估脚本：加载 checkpoint，在测试光场上算 PSNR/SSIM 并保存对比图。

用法：
  python test.py --data_source hci --data_dir data/hci/test --ckpt checkpoints/reffuse_sr_x4.pt
  python test.py --data_source synthetic --n_scenes 4 --ckpt checkpoints/reffuse_sr_x4.pt
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from config import get_config, add_common_args
from data.synthetic import make_light_field
from data.hci import load_hci_dataset
from models.ref_fuse_sr import RefFuseSR
from eval import evaluate, save_comparison


def build_test_scenes(cfg):
    if cfg.data_source == "synthetic":
        # 用不同的 seed 生成真正未见过的场景
        return [make_light_field(cfg.angular, cfg.size, cfg.dmax, seed=10000 + i)[0]
                for i in range(cfg.n_scenes)]
    return load_hci_dataset(cfg.data_dir)


def main():
    p = argparse.ArgumentParser()
    add_common_args(p)
    p.add_argument("--ckpt", type=str, default=None)
    args = p.parse_args()
    cfg = get_config(vars(args))

    ckpt_path = args.ckpt or os.path.join(cfg.ckpt_dir, f"reffuse_sr_x{cfg.scale}.pt")
    state = torch.load(ckpt_path, map_location=cfg.device)
    model = RefFuseSR(cfg.n_feat, cfg.n_blocks, cfg.scale).to(cfg.device)
    model.load_state_dict(state["state_dict"])

    lfs = build_test_scenes(cfg)
    print(f"[test] {len(lfs)} 个测试光场")

    op, os_, bp, bs = [], [], [], []
    for i, lf in enumerate(lfs):
        r = evaluate(model, lf, cfg, cfg.device)
        op.append(r["ours_psnr"]); os_.append(r["ours_ssim"])
        bp.append(r["bic_psnr"]); bs.append(r["bic_ssim"])
        print(f"  场景{i}: ours {r['ours_psnr']:.2f}/{r['ours_ssim']:.4f}  "
              f"bicubic {r['bic_psnr']:.2f}/{r['bic_ssim']:.4f}")

    print(f"\n[avg] ours    PSNR={np.mean(op):.2f}  SSIM={np.mean(os_):.4f}")
    print(f"[avg] bicubic PSNR={np.mean(bp):.2f}  SSIM={np.mean(bs):.4f}")

    out_png = os.path.join(cfg.out_dir, "test_compare.png")
    save_comparison(model, lfs[0], cfg, out_png, cfg.device)
    print(f"[img] saved -> {out_png}")


if __name__ == "__main__":
    main()
