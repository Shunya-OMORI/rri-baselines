"""radarODE の段をつなぐデータ処理: 心周期の切り方・SCEG の学習データ・形態参照(断片を PPI に合わせて伸縮・連結)・長期の学習データ.

心周期の定義(論文 p.9 "cardiac cycles are obtained by the NeuroKit2"): NeuroKit2 の ecg_segment と同じく R 波の 0.35·RR 前から
0.65·RR 後まで(nk.ecg_segment の ratio_pre = 0.35). 隣の周期と隙間なく並ぶように, 周期 i の始まりを
b_i = R_{i−1} + 0.65·(R_i − R_{i−1}) とし, 周期 i = [b_i, b_{i+1}) とする(著者のデータ生成も「次の周期の始まり = この周期の終わり」).
SCEG の入力は, その周期の R 波を中心にした 4 s(著者の data_generation.ipynb セル 17: start = R − 400)の SST.
ただし窓の外は見ないので, 評価と同じく「4 s 窓 [s, s+800) の SST」から R 波を中心に 120 フレームを切り, 窓の外は 0 にする.
学習では R 波を含む評価格子(0.2 s 刻み)の窓を毎回ランダムに選ぶ(端の拍も評価と同じ形で学ぶ).

推論での周期の位置(論文に記述なし. ppi.py の位相 φ を使う): R̂_k = φ − δ + k·PPI. δ はエネルギー波形のピークと R 波のずれ
(機械的な振動は電気的な興奮より遅れる, p.7)で, 学習側の試行から中央値で決める. 断片 k は [R̂_k − 0.35·PPI, R̂_k + 0.65·PPI) に置く.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from rri_baselines.protocol import FS, STEP, WINDOW_LEN, load_trial

RATIO_PRE = 0.35
SHAPE_LEN = 200


def cycle_bounds(peaks: np.ndarray, i: int) -> tuple[int, int]:
    """周期 i(R 波 peaks[i] を含む)の [始まり, 終わり). peaks[i−1] と peaks[i+1] が要る."""
    b0 = peaks[i - 1] + (1 - RATIO_PRE) * (peaks[i] - peaks[i - 1])
    b1 = peaks[i] + (1 - RATIO_PRE) * (peaks[i + 1] - peaks[i])
    return int(round(b0)), int(round(b1))


def to_len(x: np.ndarray, n: int = SHAPE_LEN) -> np.ndarray:
    """著者の down_sample と同じ線形補間."""
    return np.interp(np.linspace(0, len(x), n), np.arange(len(x)), x)


def minmax01(x: np.ndarray) -> np.ndarray:
    lo, hi = np.min(x), np.max(x)
    return (x - lo) / (hi - lo) if hi - lo > 1e-12 else np.zeros_like(x)


class Trials:
    """試行の RCG・ECG・R 波を 1 度だけ読む."""

    def __init__(self, trial_ids: list[int]):
        self.data = {}
        for t in trial_ids:
            tr = load_trial(t)
            self.data[t] = (np.ascontiguousarray(tr["rcg_z"].T, dtype=np.float32), tr["ecg"].astype(np.float32),
                            tr["peaks"].astype(np.int64))

    def __getitem__(self, t):
        return self.data[t]


class SCEGBeats(Dataset):
    """SCEG の学習データ: 1 拍 = 1 標本(論文 p.14: radarODE は心周期ごとの 19k 組).
    返す値: 窓の RCG (50, 800)・窓の中の R 波の位置[フレーム]・正解の断片 (1, 200)([0, 1], 公開実装の shapeLoss と同じ正規化)."""

    def __init__(self, trials: Trials, train: bool):
        self.trials, self.train, self.items = trials, train, []
        for t, (rcg, _, peaks) in trials.data.items():
            n = rcg.shape[1]
            for i in range(1, len(peaks) - 1):
                b0, b1 = cycle_bounds(peaks, i)
                if b0 < 0 or b1 > n or b1 - b0 < 20:
                    continue
                lo, hi = self._grid(peaks[i], n)
                if lo <= hi:
                    self.items.append((t, i))

    @staticmethod
    def _grid(r: int, n: int) -> tuple[int, int]:
        lo = max(0, -(-(r - WINDOW_LEN + 1) // STEP))                      # R 波を含む評価格子の窓の開始の番号
        hi = min(r // STEP, (n - WINDOW_LEN) // STEP)
        return lo, hi

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        t, i = self.items[k]
        rcg, ecg, peaks = self.trials[t]
        r = int(peaks[i])
        lo, hi = self._grid(r, rcg.shape[1])
        g = int(torch.randint(lo, hi + 1, ()).item()) if self.train else lo + (k * 7919) % (hi - lo + 1)
        s = g * STEP
        b0, b1 = cycle_bounds(peaks, i)
        target = minmax01(to_len(ecg[b0:b1]))
        center = int(round((r - s) * 30 / FS))
        return (torch.from_numpy(rcg[:, s:s + WINDOW_LEN]), torch.tensor(center),
                torch.from_numpy(target[None].astype(np.float32)))


def anchors(ppi_ms: float, phase: float, delta: float) -> np.ndarray:
    """窓の中の周期の R 波の推定位置[標本]. 周期 [R − 0.35P, R + 0.65P) が窓 [0, 800) と重なるものすべて."""
    if not (np.isfinite(ppi_ms) and np.isfinite(phase)):
        return np.zeros(0)
    p = ppi_ms / 1000.0 * FS
    r0 = phase - delta
    k = np.arange(np.floor((-(1 - RATIO_PRE) * p - r0) / p), np.ceil((WINDOW_LEN + RATIO_PRE * p - r0) / p) + 1)
    r = r0 + k * p
    return r[(r + (1 - RATIO_PRE) * p > 0) & (r - RATIO_PRE * p < WINDOW_LEN)]


def place(pieces: torch.Tensor, r_hat: np.ndarray, ppi_ms: float) -> torch.Tensor:
    """論文 p.8: "resample all the ECG pieces ... with respect to the previous PPI estimation, and ... concatenated".
    断片 (n, 200) を長さ PPI に線形補間し, [R̂ − 0.35P, R̂ + 0.65P) に置いた 800 点. 断片が無い所は 0."""
    out = torch.zeros(WINDOW_LEN, device=pieces.device)
    if len(r_hat) == 0:
        return out
    p = ppi_ms / 1000.0 * FS
    length = max(int(round(p)), 2)
    stretched = F.interpolate(pieces[:, None, :], size=length, mode="linear", align_corners=True)[:, 0]
    for piece, r in zip(stretched, r_hat):
        a = int(round(r - RATIO_PRE * p))
        lo, hi = max(a, 0), min(a + length, WINDOW_LEN)
        if lo < hi:
            out[lo:hi] = piece[lo - a:hi - a]
    return out


def mechanical_delay(phases: np.ndarray, ppis_ms: np.ndarray, starts: np.ndarray, peaks: np.ndarray) -> np.ndarray:
    """窓ごとの「位相 φ − 最も近い R 波」[標本](±P/2 に折り返す). 学習側の試行で集めて中央値を δ にする."""
    out = []
    for phi, ppi, s in zip(phases, ppis_ms, starts):
        if not (np.isfinite(phi) and np.isfinite(ppi)):
            continue
        p = ppi / 1000.0 * FS
        r = peaks[(peaks >= s) & (peaks < s + WINDOW_LEN)] - s
        if len(r) == 0:
            continue
        d = (phi - r + p / 2) % p - p / 2
        out.append(d[np.argmin(np.abs(d))])
    return np.asarray(out)


class LongWindows(Dataset):
    """長期モジュールの学習データ: (窓の RCG (50, 800), 形態参照 (1, 800), 正解 ECG (1, 800) [0, 1])."""

    def __init__(self, trials: Trials, refs: dict[int, tuple[np.ndarray, np.ndarray]]):
        self.trials, self.items, self.refs = trials, [], refs
        for t, (starts, _) in refs.items():
            self.items += [(t, j) for j in range(len(starts))]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        t, j = self.items[k]
        rcg, ecg, _ = self.trials[t]
        starts, ref = self.refs[t]
        s = int(starts[j])
        return (torch.from_numpy(rcg[:, s:s + WINDOW_LEN]), torch.from_numpy(ref[j][None].astype(np.float32)),
                torch.from_numpy(minmax01(ecg[s:s + WINDOW_LEN])[None].astype(np.float32)))
