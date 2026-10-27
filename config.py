"""路线 A（光场空间超分 + 中心参考图）复现工程的全局配置。

所有脚本共享同一份默认配置，命令行参数可覆盖任意键（仅覆盖显式传入的项）。
"""
import os
from types import SimpleNamespace

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

_DEFAULTS = dict(
    # ---- 数据 ----
    data_source="synthetic",   # synthetic | hci
    n_scenes=8,                # synthetic 场景数
    angular=5,                 # 角度分辨率 (U=V)，HCI 为 9
    size=64,                   # 高分辨率空间尺寸 (H=W)，HCI 为 512
    scale=4,                   # 超分倍率 2 / 4
    dmax=6.0,                  # 合成光场最大视差
    patch=None,                # 训练时裁剪的 HR patch 大小；None=整图
    val_split=0.25,            # 验证集比例

    # ---- 模型 ----
    n_feat=24,
    n_blocks=3,

    # ---- 训练 ----
    iters=200,
    lr=1e-3,
    device="cpu",
    seed=0,

    # ---- 路径 ----
    data_dir=os.path.join(BASE_DIR, "data", "hci"),
    ckpt_dir=os.path.join(BASE_DIR, "checkpoints"),
    out_dir=os.path.join(BASE_DIR, "results"),
)


def get_config(overrides=None):
    cfg = dict(_DEFAULTS)
    if overrides:
        for k, v in overrides.items():
            if v is not None:
                cfg[k] = v
    return SimpleNamespace(**cfg)


def add_common_args(parser):
    """把所有默认配置项注册为命令行参数（默认 None，未传则用默认值）。"""
    for k, v in _DEFAULTS.items():
        if isinstance(v, bool):
            parser.add_argument(f"--{k}", action="store_true", default=None)
        else:
            parser.add_argument(f"--{k}", type=type(v), default=None)
    return parser
