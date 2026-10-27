"""HCI 4D Light Field Benchmark 加载器与下载辅助。

官方地址：
  - https://lightfield-analysis.uni-konstanz.de/        （4D Light Field Benchmark）
  - http://hci-lightfield.iwr.uni-heidelberg.de/        （旧版 light field archive）

说明：
  - HCI 基准为 9x9 视角、每视角 512x512，HDF5(.h5) 格式。
  - 常见主数据集键名："LF"（shape (U,V,H,W,C)），旧版 archive 还含 "GT_DEPTH"。
  - 本加载器会尽量自动识别张量布局并归一化为 (U,V,H,W,C)。
"""
import os
import glob
import h5py
import numpy as np


def _normalize_lf(arr):
    """把常见光场数组布局归一化为 (U, V, H, W, C)。"""
    a = np.asarray(arr)
    if a.ndim != 5:
        raise ValueError(f"期望 5 维光场 (U,V,H,W,C)，实际 shape={a.shape}")
    s = a.shape
    is_c_last = s[-1] in (1, 3)
    is_c_first = s[0] in (1, 3)

    # (U,V,H,W,C)
    if is_c_last and s[0] <= 32 and s[1] <= 32:
        return a
    # (H,W,U,V,C)
    if is_c_last and s[2] <= 32 and s[3] <= 32:
        return np.transpose(a, (2, 3, 0, 1, 4))
    # (C,U,V,H,W)
    if is_c_first and s[1] <= 32 and s[2] <= 32:
        return np.transpose(a, (1, 2, 3, 4, 0))
    # 兜底：直接返回，交由调用方检查
    return a


def load_hci_h5(path):
    """读取单个 .h5 光场文件，返回 (U,V,H,W,C) uint8。"""
    with h5py.File(path, "r") as f:
        key = None
        for cand in ("LF", "lf", "data", "lightfield", "LightField"):
            if cand in f:
                key = cand
                break
        if key is None:
            keys = [k for k in f.keys() if isinstance(f[k], h5py.Dataset)]
            if not keys:
                raise ValueError(f"{path} 中未找到光场数据集")
            key = keys[0]
        arr = np.array(f[key])
    arr = _normalize_lf(arr)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def load_hci_dataset(data_dir):
    """加载目录下所有 .h5 光场，返回 list，每项 (U,V,H,W,C) uint8。"""
    paths = sorted(glob.glob(os.path.join(data_dir, "**", "*.h5"), recursive=True))
    if not paths:
        raise FileNotFoundError(f"{data_dir} 下没有 .h5 文件，请先下载 HCI 数据")
    lfs = [load_hci_h5(p) for p in paths]
    return lfs


# 官方页面上的各子集（下载前请到官网核对最新链接）
HCI_SOURCES = {
    "official": "https://lightfield-analysis.uni-konstanz.de/",
    "archive": "http://hci-lightfield.iwr.uni-heidelberg.de/",
}


def download_hci_scene(url, dest_path):
    """下载单个场景（用 python 标准库，避免依赖 curl/wget）。"""
    import urllib.request
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    print(f"[download] {url} -> {dest_path}")
    urllib.request.urlretrieve(url, dest_path)
    return dest_path
