"""EQ-Radio を提案手法と同じ 4 s 窓に当て, 窓の中央の時刻をまたぐ RR を 1 つ出させる.

窓の中だけを入力にする(試行全体を見ない). 4 s 窓では両端の拍が途中で切れるので, 区切りの両端に「部分区間」を許す:
左端の部分区間 x[0:l] はテンプレート(長さ Lref に伸縮)の後半 w[Lref−l:Lref] と, 右端 x[n−l:n] は前半 w[0:l] と照合する.
Lref は直前の反復の完全な区間の長さの中央値. テンプレートの更新は完全な区間だけで行う(論文の式 7).
拍の時刻は |μ| 最大の位置(テンプレートの位相 p)を各区間に写した点.

論文で決まらない点と, 相手に有利な側に置いた選び方:
- 点(50 点のどれを使うか): 冒頭 30 s で拍の形が最もそろう点(正解を使わない, results/eqradio/channels.json)
- 区間の長さの範囲と初期値: 論文既定 0.5〜1.2 s と広げた 0.4〜1.4 s, 等間隔 / 自己相関で始める 4 通りを,
  test 以外の 10 名の誤差で分割ごとに選ぶ
- 拍の時刻と R 波の一定のずれ d: ±0.5 s を 5 ms 刻みで, test 以外の 10 名の誤差が最小の値(正解を使うので相手に有利)

  .venv/bin/python -m rri_baselines.eqradio.window compute [並列数] | evaluate
"""
import json
import sys
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from numba import njit

from rri_baselines.eqradio.core import MAX_ITER, TEMPLATE_LEN, segment_costs, update_template, warp
from rri_baselines.protocol import FS, STEP, WINDOW_LEN, eval_windows, load_trial, loso_folds, score, trials_by_subject

WIN = WINDOW_LEN
CENTER = WIN // 2
DELAYS = np.arange(-100, 101)                  # 拍の時刻 − R 波[標本]. ±0.5 s を 5 ms 刻み
VARIANTS = [("paper_uniform", 0.5, 1.2, "uniform"), ("paper_acf", 0.5, 1.2, "acf"),
            ("wide_uniform", 0.4, 1.4, "uniform"), ("wide_acf", 0.4, 1.4, "acf")]
REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "results/eqradio"
CACHE = REPO / "cache/eqradio/window"
CHANNELS = {int(k): v for k, v in json.loads((OUT / "channels.json").read_text()).items()}

@njit(cache=True)
def segment_free_ends(costs, lmin, first_cost, last_cost):
    """式 6 の動的計画法に, 両端の部分区間(長さ 1〜len(first_cost)−1)を足したもの.
    戻り値: 区切り目 (m+1,) と, 各区間の種類 (m,)  0 = 完全, 1 = 左端の部分, 2 = 右端の部分."""
    n_len, n = costs.shape
    lref = len(first_cost) - 1
    dist = np.full(n + 1, np.inf)
    back = np.zeros(n + 1, dtype=np.int64)
    kind = np.zeros(n + 1, dtype=np.int64)
    dist[0] = 0.0
    for t in range(1, n + 1):
        if t <= lref:
            dist[t], back[t], kind[t] = first_cost[t], t, 1
        for i in range(n_len):
            length = lmin + i
            if length > t:
                break
            v = dist[t - length] + costs[i, t - length]
            if v < dist[t]:
                dist[t], back[t], kind[t] = v, length, 0
    best, best_l, best_kind = dist[n], 0, 0
    for l in range(1, lref + 1):
        v = dist[n - l] + last_cost[l]
        if v < best:
            best, best_l, best_kind = v, l, 2
    bounds, kinds = [n], []
    t = n
    if best_kind == 2:
        kinds.append(2)
        t -= best_l
        bounds.append(t)
    while t > 0:
        kinds.append(kind[t])
        t -= back[t]
        bounds.append(t)
    return np.array(bounds[::-1], dtype=np.int64), np.array(kinds[::-1], dtype=np.int64)


@lru_cache(maxsize=None)
def warp_matrix(m: int, lmin: int, lmax: int) -> tuple[np.ndarray, np.ndarray]:
    """ω(μ, L) = A_L μ(3 次スプラインは μ について線形). 全 L の A_L を縦に積んだ行列と, 各 L の開始行."""
    from scipy.interpolate import CubicSpline
    grid = np.linspace(0.0, 1.0, m)
    blocks = [CubicSpline(grid, np.eye(m))(np.linspace(0.0, 1.0, L)) for L in range(lmin, lmax + 1)]
    return np.vstack(blocks), np.cumsum([0] + [len(b) for b in blocks])


def fast_segment_costs(x: np.ndarray, mu: np.ndarray, lmin: int, lmax: int) -> np.ndarray:
    """eqradio.segment_costs と同じ値を, 伸縮を行列積に, 相関を一括の FFT にして計算する."""
    a, rows = warp_matrix(len(mu), lmin, lmax)
    w_all = a @ mu
    n, n_len = len(x), lmax - lmin + 1
    nfft = 1 << int(np.ceil(np.log2(n + lmax)))
    w_pad = np.zeros((n_len, nfft))
    for i in range(n_len):
        w = w_all[rows[i]:rows[i + 1]]
        w_pad[i, :len(w)] = w[::-1]
    conv = np.fft.irfft(np.fft.rfft(w_pad, axis=1) * np.fft.rfft(x, nfft)[None, :], nfft, axis=1)
    csum = np.concatenate([[0.0], np.cumsum(x * x)])
    costs = np.full((n_len, n), np.inf)
    for i in range(n_len):
        L = lmin + i
        w = w_all[rows[i]:rows[i + 1]]
        corr = conv[i, L - 1:n]                                  # Σ_j x[s+j] w[j], s = 0..n−L
        costs[i, : n - L + 1] = (csum[L:] - csum[:-L]) - 2.0 * corr + w @ w
    return costs


def edge_costs(x, mu, lref):
    w = warp(mu, lref)
    first = np.array([np.inf] + [float(np.sum((x[:l] - w[lref - l:]) ** 2)) for l in range(1, lref + 1)])
    last = np.array([np.inf] + [float(np.sum((x[len(x) - l:] - w[:l]) ** 2)) for l in range(1, lref + 1)])
    return first, last


def initial(x, lmin, lmax, init):
    if init == "uniform":
        period = (lmin + lmax) // 2
    else:
        ac = np.correlate(x, x, mode="full")[len(x) - 1:]
        period = lmin + int(np.argmax(ac[lmin:lmax + 1]))
    bounds = np.arange(0, len(x) - period + 1, period)
    return bounds, period


def segment_window(x, fs, bmin, bmax, init):
    """1 窓の拍の時刻(窓の中の標本位置)と, 完全な区間の長さの中央値(拍が 2 つ取れないときの予備の予測)."""
    lmin, lmax = int(round(bmin * fs)), int(round(bmax * fs))
    lengths = np.arange(lmin, lmax + 1)
    bounds, lref = initial(x, lmin, lmax, init)
    mu = update_template(x, bounds)
    kinds = np.zeros(len(bounds) - 1, dtype=np.int64)
    for _ in range(MAX_ITER):
        first, last = edge_costs(x, mu, lref)
        new_bounds, new_kinds = segment_free_ends(fast_segment_costs(x, mu, lmin, lmax), lmin, first, last)
        full = new_kinds == 0
        if not full.any():
            break
        converged = np.array_equal(new_bounds, bounds)
        bounds, kinds = new_bounds, new_kinds
        full_b = np.stack([bounds[:-1][full], bounds[1:][full]], 1)
        mu = sum(int(e - s) * warp(x[s:e], TEMPLATE_LEN) for s, e in full_b) / sum(int(e - s) for s, e in full_b)   # 式 7(完全な区間だけ)
        lref = int(np.median(np.diff(full_b, axis=1)))
        if converged:
            break
    p = int(np.argmax(np.abs(mu))) / (len(mu) - 1)
    beats = []
    for (s, e), k in zip(zip(bounds[:-1], bounds[1:]), kinds):
        l = e - s
        if k == 0:
            beats.append(s + p * l)
        elif k == 1 and p * lref >= lref - l:          # 左端の部分区間がテンプレートの位相 p を含む
            beats.append(p * lref - (lref - l))
        elif k == 2 and p * lref < l:                  # 右端の部分区間が位相 p を含む
            beats.append(s + p * lref)
    full_len = np.diff(bounds)[kinds == 0]
    return np.array(beats), (float(np.median(full_len)) if len(full_len) else float(lref))

def compute_trial(trial_id: int) -> str:
    path = CACHE / f"trial{trial_id}.npz"
    if path.exists():
        return f"trial {trial_id}: skip"
    trial = load_trial(trial_id)
    x_all = trial["rcg_z"][:, CHANNELS[trial_id]].astype(np.float64)
    starts, _ = eval_windows(len(x_all), trial["peaks"])
    out = {"starts": starts}
    for name, bmin, bmax, init in VARIANTS:
        flat, offs, fallback = [], [0], []
        for s in starts:
            beats, period = segment_window(x_all[s:s + WIN], FS, bmin, bmax, init)
            flat.extend(beats)
            offs.append(len(flat))
            fallback.append(period)
        out[f"beats_{name}"], out[f"offs_{name}"], out[f"fallback_{name}"] = np.array(flat), np.array(offs), np.array(fallback)
    np.savez(path, **out)
    return f"trial {trial_id}: {len(starts)} 窓"


@lru_cache(maxsize=None)
def predictions(trial_id: int, name: str) -> tuple[np.ndarray, np.ndarray]:
    """(ずれ d の数, 窓の数) の予測 RR[ms] と, 予備の予測(完全な区間の長さの中央値)に落ちた窓."""
    arr = np.load(CACHE / f"trial{trial_id}.npz")
    beats, offs, fallback = arr[f"beats_{name}"], arr[f"offs_{name}"], arr[f"fallback_{name}"]
    pred = np.empty((len(DELAYS), len(offs) - 1))
    used_fb = np.zeros((len(DELAYS), len(offs) - 1), dtype=bool)
    at = CENTER + DELAYS
    for w in range(len(offs) - 1):
        b = beats[offs[w]:offs[w + 1]]
        k = np.searchsorted(b, at, side="right")
        ok = (k > 0) & (k < len(b))
        kk = np.clip(k, 1, max(len(b) - 1, 1))
        pred[:, w] = np.where(ok, (b[kk] - b[kk - 1]) if len(b) >= 2 else fallback[w], fallback[w])
        used_fb[:, w] = ~ok
    return pred / FS * 1000.0, used_fb


@lru_cache(maxsize=None)
def true_ms(trial_id: int) -> np.ndarray:
    trial = load_trial(trial_id)
    return eval_windows(len(trial["rcg_z"]), trial["peaks"])[1]


def trial_mae(trial_id: int, name: str) -> np.ndarray:
    pred, _ = predictions(trial_id, name)
    y = true_ms(trial_id)
    ok = np.isfinite(y)
    return np.mean(np.abs(pred[:, ok] - y[ok][None, :]), axis=1)


def evaluate() -> dict:
    by_subject = trials_by_subject()
    names = [v[0] for v in VARIANTS]
    preds, folds = {}, []
    for fold in loso_folds():
        fit_ids = [t for s in fold.train + fold.val for t in by_subject[s]]
        fit = {n: np.mean([trial_mae(t, n) for t in fit_ids], axis=0) for n in names}
        chosen = min(names, key=lambda n: fit[n].min())
        i = int(np.argmin(fit[chosen]))
        for t in by_subject[fold.test[0]]:
            preds[t] = predictions(t, chosen)[0][i]
        folds.append({"test": fold.test[0], "variant": chosen, "delay_ms": int(DELAYS[i]) * 5})
    result = {**score(preds), "folds": folds}
    (OUT / "evaluation.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    return result


def main() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    if sys.argv[1] == "compute":
        ids = [t for ts in trials_by_subject().values() for t in ts]
        with Pool(int(sys.argv[2]) if len(sys.argv) > 2 else 8) as pool:
            for line in pool.imap_unordered(compute_trial, ids):
                print(line, flush=True)
    else:
        r = evaluate()
        print(f"EQ-Radio(同じ 4 s 窓・除外なし): 参加者平均 {r['mae_ms']:.3f} ms")


if __name__ == "__main__":
    main()
