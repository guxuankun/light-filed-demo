"""一键 demo：合成光场 + CPU 小规模训练 + 评估 + 对比图。

运行：python run_demo.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import get_config
from train import build_scenes, train_model
from eval import evaluate, save_comparison


def main():
    cfg = get_config(dict(
        data_source="synthetic", n_scenes=8, angular=5, size=64, scale=4,
        dmax=6.0, n_feat=32, n_blocks=4, iters=400, lr=1e-3,
        device="cpu", seed=0,
    ))
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    os.makedirs(cfg.out_dir, exist_ok=True)

    lfs = build_scenes(cfg)
    n_val = 2
    lfs_val, lfs_train = lfs[:n_val], lfs[n_val:]
    print(f"[demo] 合成光场: train={len(lfs_train)} val={len(lfs_val)}  "
          f"angular={cfg.angular} size={cfg.size} scale={cfg.scale}")

    model = train_model(cfg, lfs_train, lfs_val)

    print("\n[demo] 验证集全视角平均指标:")
    for i, lf in enumerate(lfs_val):
        r = evaluate(model, lf, cfg, cfg.device)
        print(f"  场景{i}: ours PSNR={r['ours_psnr']:.2f}/SSIM={r['ours_ssim']:.4f}  "
              f"vs  bicubic PSNR={r['bic_psnr']:.2f}/SSIM={r['bic_ssim']:.4f}")

    out_png = os.path.join(cfg.out_dir, "demo_compare_x4.png")
    save_comparison(model, lfs_val[0], cfg, out_png, cfg.device)
    print(f"\n[demo] 对比图已保存: {out_png}")


if __name__ == "__main__":
    main()
