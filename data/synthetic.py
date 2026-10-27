"""合成光场生成器（离线可用）。

生成一个带真实视差的合成光场：先生成一张含高频细节的纹理图 + 一张平滑视差场，
再把纹理按「(u-uc)*disp, (v-vc)*disp」平移得到每个视角。
中心视角 (uc, vc) 无平移，等于原纹理，正好充当路线 A 的「高清中心参考图」。
"""
import numpy as np
import cv2


def _smooth_disp(h, w, n_blobs, rng, dmax):
    """平滑视差场：近物视差大（值大），远物视差小。"""
    d = np.zeros((h, w), dtype=np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    for _ in range(n_blobs):
        cx, cy = rng.integers(0, w), rng.integers(0, h)
        r = float(rng.integers(8, max(12, h // 3)))
        amp = float(rng.uniform(0.4, 1.0))
        d += amp * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r * r))
    d = d / (d.max() + 1e-8) * dmax
    return d


def _texture(h, w, rng):
    """含结构化高频细节的彩色纹理图。

    关键：细节必须是「可预测的结构」（锐利边缘、细网格），而非随机噪声，
    这样参考图里的高频信息才能被网络学来转移到其他视角。
    """
    img = np.zeros((h, w, 3), dtype=np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    img[..., 0] = 110 + 90 * (xx / w)
    img[..., 1] = 120 + 80 * (yy / h)
    img[..., 2] = 150

    # 细网格纹理（高频结构）
    cell = 4
    chk = ((xx // cell + yy // cell) % 2).astype(np.float32)
    img += (chk[..., None] - 0.5) * 30

    # 大量高对比小形状（锐利边缘，下采样后会丢失）
    for _ in range(120):
        cx, cy = rng.integers(0, w), rng.integers(0, h)
        s = int(rng.integers(1, 6))
        color = rng.uniform(0, 255, size=3)
        if rng.random() < 0.5:
            cv2.rectangle(img, (cx, cy), (min(w, cx + s), min(h, cy + s)), color, -1)
        else:
            cv2.circle(img, (cx, cy), s, color, -1)

    # 轻微噪声，避免主导高频成分
    img += rng.uniform(-6, 6, size=(h, w, 1))
    return np.clip(img, 0, 255).astype(np.uint8)


def make_light_field(angular=5, h=64, dmax=6.0, seed=0):
    """生成合成光场。

    返回 (lf, disp)：
      lf    : (angular, angular, h, w, 3) uint8
      disp  : (h, w) float32 视差场（可选，仅用于可视化/调试）
    """
    rng = np.random.default_rng(seed)
    tex = _texture(h, h, rng).astype(np.float32)
    disp = _smooth_disp(h, h, 5, rng, dmax)
    c = angular // 2
    yy, xx = np.mgrid[0:h, 0:h].astype(np.float32)
    lf = np.zeros((angular, angular, h, h, 3), dtype=np.uint8)
    for u in range(angular):
        for v in range(angular):
            map_x = xx + (u - c) * disp
            map_y = yy + (v - c) * disp
            view = cv2.remap(tex, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT)
            lf[u, v] = np.clip(view, 0, 255).astype(np.uint8)
    return lf, disp
