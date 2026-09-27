"""radarODE-MTL の学習データを, 提案手法と同じ窓・目標量で作る.

公開実装の 1 標本(spectrum_dataset.SpectrumECGDataset.__getitem__)と同じ形を返す:
  SST (50, 71, 120)・ECG_shape (1, 200)・PPI (1, 260)・Anchor (1, 800)
違いは「どの拍を 1 拍とするか」だけで, ここでは窓の中央をまたぐ 2 つの R 波の間(提案手法の目標量)とする.
  ECG_shape: その拍の ECG を 200 点に線形補間(公開実装の down_sample)
  PPI      : その拍の ECG を長さ 260 まで −10 で埋めたもの(損失は長さ L をクラス L−1 と読む)
  Anchor   : 窓の中の R 波の位置に 1 を立てた 800 点(公開実装の spike 形式)
SST は試行全体の SST(sst.py)を 30 Hz で切り出す. 窓の開始 s[200 Hz 標本] はフレーム s·30/200.
"""
import numpy as np
import torch
from torch.utils.data import Dataset

from rri_baselines.protocol import FS, WINDOW_LEN, load_trial, rr_spanning
from rri_baselines.radarode.sst import CACHE

PPI_LEN, SHAPE_LEN, SST_FRAMES = 260, 200, 120


def down_sample(x: np.ndarray, n: int = SHAPE_LEN) -> np.ndarray:
    return np.interp(np.linspace(0, len(x), n), np.arange(len(x)), x)


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
                rr = rr_spanning(tr["peaks"], s + WINDOW_LEN / 2.0, s, s + WINDOW_LEN)
                if labeled_only and not np.isfinite(rr):
                    continue
                self.items.append((t, s))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        t, s = self.items[i]
        sst, ecg, peaks = self.trials[t]
        f0 = s * 30 // FS
        x = torch.from_numpy(np.asarray(sst[:, :, f0:f0 + SST_FRAMES], dtype=np.float32))
        k = int(np.searchsorted(peaks, s + WINDOW_LEN / 2.0))
        piece = ecg[peaks[k - 1]:peaks[k]] if 0 < k < len(peaks) else ecg[s:s + 1]
        ppi = np.full(PPI_LEN, -10.0, dtype=np.float32)
        ppi[:min(len(piece), PPI_LEN)] = piece[:PPI_LEN]
        anchor = np.zeros(WINDOW_LEN, dtype=np.float32)
        inside = peaks[(peaks >= s) & (peaks < s + WINDOW_LEN)] - s
        anchor[inside] = 1.0
        return x, {"ECG_shape": torch.from_numpy(down_sample(piece)[None].astype(np.float32)),
                   "PPI": torch.from_numpy(ppi[None]), "Anchor": torch.from_numpy(anchor[None])}
