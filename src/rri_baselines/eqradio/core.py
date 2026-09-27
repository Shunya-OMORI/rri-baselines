"""EQ-Radio の拍の区切り (Zhao, Adib, Katabi, MobiCom 2016, §5.2–5.3, Algorithm 1) の再現.

加速度の系列 x を, 長さ bmin〜bmax の区間に分け, 各区間を線形伸縮(3 次スプライン)したテンプレート μ との
二乗誤差の和を最小にする. 区切り S の更新(動的計画法, 式 6)とテンプレート μ の更新(重み付き平均, 式 7)を交互に回す.

論文で決まらない点の扱い(README に理由):
- 初期値: 論文は μ0 = 0 だが, そのときはどの区切りも同じコストになり一意に決まらない.
  "uniform" は (bmin+bmax)/2 の等間隔, "acf" は自己相関の最大の極大の周期で等間隔に切った区切りから始める.
- 拍の時刻: 区切り目の心周期上の位置は任意なので, テンプレートの |μ| 最大の位置を各区間に写した時刻を拍の時刻とする.
"""
import numpy as np
from numba import njit
from scipy.interpolate import CubicSpline
from scipy.signal import fftconvolve

TEMPLATE_LEN = 170      # m. 論文に値がない. 0.85 s × 200 Hz(bmin・bmax の中点)
MAX_ITER = 30           # 論文では平均 8 回, 最大 16 回で収束


def warp(v: np.ndarray, length: int) -> np.ndarray:
    """ω(v, L): 3 次スプラインで長さ L に線形伸縮する(論文の脚注 3)."""
    return CubicSpline(np.linspace(0.0, 1.0, len(v)), v)(np.linspace(0.0, 1.0, length))


def segment_costs(x: np.ndarray, mu: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """C[i, s] = ||x[s:s+L_i] − ω(μ, L_i)||². 収まらない開始位置は inf."""
    n = len(x)
    csum = np.concatenate([[0.0], np.cumsum(x * x)])
    costs = np.full((len(lengths), n), np.inf)
    for i, length in enumerate(lengths):
        w = warp(mu, length)
        corr = fftconvolve(x, w[::-1], mode="valid")            # Σ_j x[s+j] w[j], 長さ n−L+1
        costs[i, : n - length + 1] = (csum[length:] - csum[:-length]) - 2.0 * corr + w @ w
    return costs


@njit(cache=True)
def best_segmentation(costs: np.ndarray, lmin: int) -> np.ndarray:
    """式 6 の動的計画法. D_t = min_L {D_{t−L} + C[L, t−L]}. 戻り値は区切り目 [0, …, n]."""
    n_len, n = costs.shape
    dist = np.full(n + 1, np.inf)
    back = np.zeros(n + 1, dtype=np.int64)
    dist[0] = 0.0
    for t in range(lmin, n + 1):
        best, best_len = np.inf, -1
        for i in range(n_len):
            length = lmin + i
            if length > t:
                break
            v = dist[t - length] + costs[i, t - length]
            if v < best:
                best, best_len = v, length
        dist[t], back[t] = best, best_len
    if not np.isfinite(dist[n]):
        return np.empty(0, dtype=np.int64)
    out = [n]
    t = n
    while t > 0:
        t -= back[t]
        out.append(t)
    return np.array(out[::-1], dtype=np.int64)


def update_template(x: np.ndarray, bounds: np.ndarray, m: int = TEMPLATE_LEN) -> np.ndarray:
    """式 7: μ = Σ |s_i| ω(s_i, m) / Σ |s_i|."""
    lengths = np.diff(bounds)
    acc = np.zeros(m)
    for start, length in zip(bounds[:-1], lengths):
        acc += length * warp(x[start:start + length], m)
    return acc / lengths.sum()


def initial_bounds(x: np.ndarray, lmin: int, lmax: int, init: str) -> np.ndarray:
    """初期の区切り. 論文の μ0 = 0 の代わり(モジュールの説明を参照)."""
    if init == "uniform":
        period = (lmin + lmax) // 2
    elif init == "acf":
        ac = np.correlate(x, x, mode="full")[len(x) - 1:]
        period = lmin + int(np.argmax(ac[lmin:lmax + 1]))
    else:
        raise ValueError(init)
    n = len(x)
    bounds = list(range(0, n - period + 1, period))
    bounds[-1] = n                                          # 端数は最後の区間に含める
    return np.array(bounds, dtype=np.int64)


def segment_heartbeats(x: np.ndarray, fs: int, bmin: float, bmax: float, init: str) -> dict:
    """Algorithm 1. 区切り・テンプレート・正規化した残差(コスト / Σx²)・反復回数."""
    lmin, lmax = int(round(bmin * fs)), int(round(bmax * fs))
    lengths = np.arange(lmin, lmax + 1)
    bounds = initial_bounds(x, lmin, lmax, init)
    mu = update_template(x, bounds)
    for it in range(1, MAX_ITER + 1):
        costs = segment_costs(x, mu, lengths)
        new_bounds = best_segmentation(costs, lmin)
        if len(new_bounds) == 0:
            break
        converged = len(new_bounds) == len(bounds) and np.array_equal(new_bounds, bounds)
        bounds = new_bounds
        mu = update_template(x, bounds)
        if converged:
            break
    resid = sum(float(np.sum((x[s:s + l] - warp(mu, l)) ** 2)) for s, l in zip(bounds[:-1], np.diff(bounds)))
    return {"bounds": bounds, "template": mu, "resid_norm": resid / float(x @ x), "n_iter": it}


def beat_times(bounds: np.ndarray, template: np.ndarray) -> np.ndarray:
    """各区間で |μ| 最大の位置に当たる時刻[標本, 実数]. R 波の時刻に対応させる拍の基準点."""
    phase = int(np.argmax(np.abs(template))) / (len(template) - 1)
    return bounds[:-1] + phase * np.diff(bounds)
