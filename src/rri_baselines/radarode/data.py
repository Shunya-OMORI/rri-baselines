"""radarODE-MTL の学習データを, 提案手法と同じ窓・目標量で作る.

公開実装の 1 標本(spectrum_dataset.SpectrumECGDataset.__getitem__)と同じ形を返す:
  SST (50, 71, 120)・ECG_shape (1, 200)・PPI (1, 260)・Anchor (1, 800)
違いは「どの拍を 1 拍とするか」だけで, ここでは窓の中央をまたぐ 2 つの R 波の間(提案手法の目標量)とする.
  ECG_shape: その拍の ECG を 200 点に線形補間(公開実装の down_sample)
  PPI      : その拍の ECG を長さ 260 まで −10 で埋めたもの(損失は長さ L をクラス L−1 と読む)
  Anchor   : 窓の中の R 波の位置に 1 を立てた 800 点(公開実装の spike 形式)
SST は試行全体の SST を 30 Hz で切り出し, 窓ごと・点ごとに min-max で [0, 1] にする. 窓の開始 s[200 Hz 標本] はフレーム s·30/200.

SST の作り方は環境変数 RADARODE_SST で選ぶ(2026-09-30. 既定は wsst).
  wsst : wsst.py. MATLAB wsst の移植で, 著者のデータ生成(CFT-RFcardi の data_generation.ipynb)と同じ 0.72〜100 Hz・行 0 が 100 Hz
  low  : sst.py. ssqueezepy の 0.2〜7.7 Hz(09-28 の 24.18 ms はこちら. 行の並びの取り違えで低い側の半分を取っていた)
対象の拍は RADARODE_TARGET で選ぶ.
  center: 窓の中央をまたぐ拍(提案手法の目標量. 論文の "the middle cardiac cycle")
  first : 窓の中で最初の R 波から次の R 波まで(著者のデータ生成のコードが実際に選ぶ拍. 窓の開始が 400 標本以降のとき)
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset

from rri_baselines.protocol import FS, WINDOW_LEN, load_trial, rr_spanning
from rri_baselines.radarode import sst as _low
from rri_baselines.radarode import wsst as _wsst

SST_VARIANT = os.environ.get("RADARODE_SST", "wsst")
TARGET = os.environ.get("RADARODE_TARGET", "center")
CACHE = {"low": _low.CACHE, "wsst": _wsst.CACHE}[SST_VARIANT]
PPI_LEN, SHAPE_LEN, SST_FRAMES = 260, 200, 120


def down_sample(x: np.ndarray, n: int = SHAPE_LEN) -> np.ndarray:
    return np.interp(np.linspace(0, len(x), n), np.arange(len(x)), x)


def target_index(peaks: np.ndarray, s: int) -> int:
    """対象の拍を peaks[k − 1]〜peaks[k] とする k."""
    if TARGET == "first":
        return int(np.searchsorted(peaks, s, side="right")) + 1          # 窓の中で最初の R 波(> s)が peaks[k − 1]
    return int(np.searchsorted(peaks, s + WINDOW_LEN / 2.0))


def target_rr(peaks: np.ndarray, s: int) -> float:
    """対象の拍の RR[標本]. 両端の R 波が窓 [s, s + 800) の外なら NaN."""
    if TARGET == "first":
        k = target_index(peaks, s)
        return float(peaks[k] - peaks[k - 1]) if k < len(peaks) and peaks[k] < s + WINDOW_LEN else np.nan
    return rr_spanning(peaks, s + WINDOW_LEN / 2.0, s, s + WINDOW_LEN)


class CenterBeatWindows(Dataset):
    def __init__(self, trial_ids: list[int], stride: int, labeled_only: bool = True) -> None:
        self.trials, self.items = {}, []
        for t in trial_ids:
            tr = load_trial(t)
            sst = np.load(CACHE / f"trial{t}.npy", mmap_mode="r")
            self.trials[t] = (sst, tr["ecg"].astype(np.float32), tr["peaks"].astype(np.int64))
            n_frames = sst.shape[-1]
            for s in range(0, len(tr["rcg_z"]) - WINDOW_LEN + 1, stride):
                if s * 30 // FS + SST_FRAMES > n_frames:
                    break
                if labeled_only and not np.isfinite(target_rr(tr["peaks"], s)):
                    continue
                self.items.append((t, s))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        t, s = self.items[i]
        sst, ecg, peaks = self.trials[t]
        f0 = s * 30 // FS
        x = np.asarray(sst[:, :, f0:f0 + SST_FRAMES], dtype=np.float32)
        lo, hi = x.min(axis=(1, 2), keepdims=True), x.max(axis=(1, 2), keepdims=True)
        x = torch.from_numpy((x - lo) / np.where(hi - lo < 1e-12, 1.0, hi - lo))        # 窓ごと・点ごとに [0, 1](著者の sst_norm_01)
        k = target_index(peaks, s)
        piece = ecg[peaks[k - 1]:peaks[k]] if 0 < k < len(peaks) else ecg[s:s + 1]
        ppi = np.full(PPI_LEN, -10.0, dtype=np.float32)
        ppi[:min(len(piece), PPI_LEN)] = piece[:PPI_LEN]
        anchor = np.zeros(WINDOW_LEN, dtype=np.float32)
        inside = peaks[(peaks >= s) & (peaks < s + WINDOW_LEN)] - s
        anchor[inside] = 1.0
        return x, {"ECG_shape": torch.from_numpy(down_sample(piece)[None].astype(np.float32)),
                   "PPI": torch.from_numpy(ppi[None]), "Anchor": torch.from_numpy(anchor[None])}
