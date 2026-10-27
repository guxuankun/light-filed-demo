# 光场空间超分复现（路线 A：高清中心参考图）

复现目标：**用公开光场数据集，把所有视角下采样成 LR，额外保留中心视角的高清原图作"参考图"，训练网络学会「HR 中心参考 + LR 其他视角 → HR 其他视角」**。所有视角都有真值，可正常监督训练。

## 原理速览

1. **光场** = 同一场景的一叠视角图（子孔径图像 SAI），形状 `(U, V, H, W, C)`。
2. **降质模型**：每个视角 bicubic 下采样 s 倍 → LR；中心视角额外保留 HR 作参考图。
3. **训练对**：`{HR 中心参考 + LR 全部视角} → {HR 全部视角}`，L1 损失监督。
4. **几何基础**：其他视角与中心视角只差一个视差（disparity）偏移，网络学习把参考图的高频细节按视差对齐、搬运到每个低清视角（本质是 Reference-based SR）。

## 目录结构

```
lf-sr/
├── config.py              # 全局配置（角度分辨率/尺寸/倍率/模型超参/路径）
├── data/
│   ├── synthetic.py       # 合成光场生成器（离线可用，含真实视差）
│   ├── hci.py             # HCI 4D Benchmark (.h5) 加载器 + 下载辅助
│   └── dataset.py         # LFSRDataset：构造 LR / 中心参考 / HR 训练对
├── models/
│   └── ref_fuse_sr.py     # RefFuseSR（含参考图） + PlainSR（无参考对照）
├── utils/metrics.py       # PSNR / SSIM
├── eval.py                # 全视角评估 + bicubic/ours/GT 对比图
├── train.py               # 训练脚本
├── test.py                # 评估脚本（加载 checkpoint）
├── compare_ref.py         # A/B 对照：参考图 vs 无参考
├── run_demo.py            # 一键 CPU demo
└── requirements.txt
```

## 快速开始（CPU 即可）

```bash
python run_demo.py        # 合成光场 + 小规模训练 + 评估 + 保存对比图
python compare_ref.py     # A/B 对照：验证参考图的价值
```

依赖：`torch`, `numpy`, `opencv-python`, `scipy`, `h5py`。

## 用真实 HCI 数据训练

1. 到 [HCI 4D Light Field Benchmark](https://lightfield-analysis.uni-konstanz.de/)
   （或旧版 [archive](http://hci-lightfield.iwr.uni-heidelberg.de/)）下载
   `training/`（28 场景）与 `additional/`（16 场景）的 `.h5` 文件；
2. 放到 `lf-sr/data/hci/` 下（支持子目录）；
3. 训练：

```bash
python train.py --data_source hci --angular 9 --size 512 --patch 64 --iters 200000 --device cuda
python test.py  --data_source hci --data_dir data/hci/additional --ckpt checkpoints/reffuse_sr_x4.pt
```

> `.h5` 主数据集键名通常为 `LF`，形状 `(U,V,H,W,C)`；加载器会自动识别常见布局。

## GPU 扩展说明

代码天然支持 GPU（`--device cuda`）。要达到论文级效果，建议在此基础上：

1. **加大模型**：`--n_feat 64 --n_blocks 8`（或换成显式视差对齐模块：可变形卷积 / disparity 引导的 warp）；
2. **patch 训练**：`--patch 64`，随机裁剪 HR patch，等效数据量大幅增加；
3. **数据增强**：随机翻转/旋转（可在 `LFSRDataset.__getitem__` 中加）；
4. **学习率调度**：余弦退火或按 iteration 衰减；
5. **评估协议**：按 LF SR 惯例报 2x/4x、全视角平均 PSNR/SSIM（本工程已实现）。

## 已知限制与诚实说明

- 本工程是**最小可复现基线**：模型很小（~66k 参数）、无显式视差对齐模块，
  在 CPU 少迭代下参考图带来 **+0.13 dB** 的一致提升（方向正确、幅度有限）。
  要拿到论文级增益，需要 GPU + 更大模型 + 显式对齐 + 全量数据训练。
- 合成光场的视差是「平移场」近似，真实光场存在遮挡/非朗伯面等复杂情况。
- 中心视角本身在训练中可被网络直接从参考图复制（恒等映射），属预期行为。

## 后续可加的实验

- 2x / 4x 倍率对比（`--scale 2`）
- 增大 `dmax` 观察参考图收益随视差的变化
- 换用 EPI / 4D 卷积 / 可变形对齐等更强结构
