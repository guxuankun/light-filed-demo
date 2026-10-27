"""评估指标：PSNR 与 SSIM（numpy 实现，无额外依赖）。"""
import numpy as np
from scipy.ndimage import uniform_filter


def psnr(a, b, max_val=1.0):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10(max_val * max_val / mse)


def _ssim_gray(a, b, max_val=1.0):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    K1, K2 = 0.01, 0.03
    C1 = (K1 * max_val) ** 2
    C2 = (K2 * max_val) ** 2
    mu1 = uniform_filter(a, size=11, mode="reflect")
    mu2 = uniform_filter(b, size=11, mode="reflect")
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu12 = mu1 * mu2
    s1 = uniform_filter(a ** 2, size=11, mode="reflect") - mu1_sq
    s2 = uniform_filter(b ** 2, size=11, mode="reflect") - mu2_sq
    s12 = uniform_filter(a * b, size=11, mode="reflect") - mu12
    m = ((2 * mu12 + C1) * (2 * s12 + C2)) / ((mu1_sq + mu2_sq + C1) * (s1 + s2 + C2))
    return float(m.mean())


def ssim(a, b, max_val=1.0):
    a = np.asarray(a)
    b = np.asarray(b)
    if a.ndim == 3:
        return float(np.mean([_ssim_gray(a[..., i], b[..., i], max_val)
                              for i in range(a.shape[-1])]))
    return _ssim_gray(a, b, max_val)
