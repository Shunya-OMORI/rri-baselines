"""MMECG の試行から mm-JEPA-ECG の入力を作る(公式の npy 形式 rcg / ecg / pos / y / hr を load_trial から作る).

公式の npy は配られていない(README「dataset … undergoing … anonymization」)ので, 形だけ公式に合わせる:
  rcg (B, 16, 50, 32)  … 512 標本 × 50 点の窓を 32 標本のパッチ 16 個に分ける(公式 create_patch_batch → reorder_rcg)
  ecg (B, 512)         … 窓ごとに最小・最大で [-1, 1] へ(DDIM の clip_denoised=True が [-1, 1] を前提にするため)
  pos (B, 50, 3)       … MMECG の posxyz(50 点の 3 次元座標[m]). 公式 space_proj の入力
  hr  (B,)             … 擬似心拍数[bpm]. 論文 p.38965: レーダを 0.18–2.5 Hz で帯域通過してピーク検出
レーダは protocol の rcg_z(試行ごとの z 標準化. 他の比較手法と同じ入力)をそのまま使う.
"""
import warnings
from pathlib import Path

import numpy as np
import torch
from scipy.signal import butter, find_peaks, peak_prominences, sosfiltfilt

from rri_baselines.protocol import FS, load_trial

SEG = 512                    # モデルの入力長(2.56 s)
N_PATCH, PATCH = 16, 32
TRAIN_STEP = 128             # 学習窓のずらし幅(論文に記載なし. 公式の log-every=155 ≒ 1 エポックの段数に近くなる値)
HR_BAND = (0.18, 2.5)        # 論文 p.38965
# ピーク検出の細部(最小間隔・卓立度)は論文に記載なし. **分割ごとに, その分割の学習 9 名の試行だけ**で, 正解 HR との相関が
# 最も高い組を選ぶ(select_hr_params). 2026-10-02 のレビューで, 以前の「分割 0 の学習 9 名で選んだ組を全分割に使う」は
# 他の分割の test 参加者の ECG に触れていたので改めた. 以下は既定値(書き直し版の初回の選択. 分割 0 の学習側での最良,
# 相関 0.25・MAE 8.9 bpm. 定数予測は 7.6 bpm)
HR_MIN_DIST = int(0.5 * FS)             # ピークの最小間隔 100 標本
HR_PROM = 0.3                           # 窓の中で最大の卓立度の 0.3 倍未満の山は捨てる
HR_GRID = [(int(d * FS), pr) for d in (0.4, 0.5, 0.6) for pr in (0.0, 0.3, 0.5)]   # 選択の候補 9 通り
CACHE = Path(__file__).resolve().parents[3] / "cache/mmjepa"


def band_filtered(rcg: np.ndarray) -> np.ndarray:
    """試行全体を 0.18–2.5 Hz の 4 次 Butterworth でゼロ位相に通す (T, 50)(次数・ゼロ位相は論文に記載なし)."""
    sos = butter(4, HR_BAND, btype="band", fs=FS, output="sos")
    return sosfiltfilt(sos, rcg.astype(np.float64), axis=0)


def _hr_of(seg: np.ndarray, dist: int = HR_MIN_DIST, prom: float = HR_PROM) -> float:
    p, _ = find_peaks(seg, distance=dist)
    if len(p) and prom > 0:
        pr = peak_prominences(seg, p)[0]
        p = p[pr >= prom * pr.max()]
    return 60.0 * FS / float(np.mean(np.diff(p))) if len(p) >= 2 else np.nan


def pseudo_hr(filtered: np.ndarray, starts: np.ndarray, dist: int = HR_MIN_DIST, prom: float = HR_PROM) -> np.ndarray:
    """窓 [s, s+512) ごとの擬似心拍数[bpm] (n,). 50 点それぞれでピーク間隔の平均から心拍数を出し, 点の中央値.
    取れない窓は試行の擬似心拍数の中央値で埋める(公式は全窓に hr を持つ)."""
    out = np.full(len(starts), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)          # 全点で山が取れない窓の All-NaN 警告
        for i, s in enumerate(starts):
            out[i] = np.nanmedian([_hr_of(filtered[s:s + SEG, c], dist, prom) for c in range(filtered.shape[1])] + [np.nan])
    fill = np.nanmedian(out) if np.isfinite(out).any() else 75.0
    return np.where(np.isfinite(out), out, fill).astype(np.float32)


def trial_hr_labels(trial_id: int, starts: np.ndarray, dist: int = HR_MIN_DIST, prom: float = HR_PROM) -> np.ndarray:
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"hr_d{dist}_p{prom}_step{int(starts[1] - starts[0]) if len(starts) > 1 else 0}_trial{trial_id}.npy"
    if f.exists():
        v = np.load(f)
        if len(v) == len(starts):
            return v
    v = pseudo_hr(band_filtered(load_trial(trial_id)["rcg_z"]), starts, dist, prom)
    np.save(f, v)
    return v


def select_hr_params(train_trials: list[int], step: int = 4 * TRAIN_STEP) -> dict:
    """擬似心拍数のピーク検出の (最小間隔, 卓立度) を, 学習側の試行だけで選ぶ(正解 R 波の窓平均 HR との相関が最大の組).
    窓は 512 標本(ずらし 4 × 128 標本)で間引く. test・val の参加者の ECG には触れない."""
    data = []
    for t in train_trials:
        tr = load_trial(t)
        st = np.arange(0, len(tr["rcg_z"]) - SEG + 1, step)
        data.append((band_filtered(tr["rcg_z"]), st, true_hr(tr["peaks"], st)))
    table = {}
    for dist, prom in HR_GRID:
        ps = np.concatenate([pseudo_hr(f, st, dist, prom) for f, st, _ in data])
        ts = np.concatenate([h for _, _, h in data])
        ok = np.isfinite(ts)
        table[f"d{dist}_p{prom}"] = {"dist": dist, "prom": prom, "corr": float(np.corrcoef(ps[ok], ts[ok])[0, 1]),
                                     "mae_bpm": float(np.mean(np.abs(ps[ok] - ts[ok])))}
    ts_all = np.concatenate([h for _, _, h in data]); ok = np.isfinite(ts_all)
    best = max(table, key=lambda k: table[k]["corr"])
    return {"best": best, "dist": table[best]["dist"], "prom": table[best]["prom"], "table": table,
            "n_windows": int(ok.sum()), "const_mae_bpm": float(np.mean(np.abs(ts_all[ok] - ts_all[ok].mean()))),
            "trials": train_trials}


def true_hr(peaks: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """正解 R 波から, 窓の中の平均心拍数[bpm](擬似ラベルの質を確かめるためだけに使う)."""
    out = np.full(len(starts), np.nan)
    for i, s in enumerate(starts):
        p = peaks[(peaks >= s) & (peaks < s + SEG)]
        if len(p) >= 2:
            out[i] = 60.0 * FS / np.mean(np.diff(p))
    return out


class WindowSet:
    """試行をつないだ信号から 512 標本の窓を切り出す. 窓は (試行の中の開始位置) で持ち, 取り出しは添字演算で一括に行う."""

    def __init__(self, trial_ids: list[int], step: int = TRAIN_STEP, device="cpu", with_hr: bool = True):
        self.device = torch.device(device)
        rcg, ecg, pos, starts, tidx, hr = [], [], [], [], [], []
        offset = 0
        for k, t in enumerate(trial_ids):
            tr = load_trial(t)
            n = len(tr["rcg_z"])
            st = np.arange(0, n - SEG + 1, step)
            rcg.append(tr["rcg_z"].astype(np.float32))
            ecg.append(tr["ecg"].astype(np.float32))
            pos.append(tr["posxyz"].astype(np.float32))
            starts.append(st + offset)
            tidx.append(np.full(len(st), k))
            if with_hr:
                hr.append(trial_hr_labels(t, st))
            offset += n
        self.rcg = torch.from_numpy(np.concatenate(rcg)).to(self.device)
        self.ecg = torch.from_numpy(np.concatenate(ecg)).to(self.device)
        self.pos = torch.from_numpy(np.stack(pos)).to(self.device)
        self.starts = torch.from_numpy(np.concatenate(starts)).to(self.device)
        self.tidx = torch.from_numpy(np.concatenate(tidx)).to(self.device)
        self.hr = torch.from_numpy(np.concatenate(hr)).to(self.device) if with_hr else None
        self._ar = torch.arange(SEG, device=self.device)

    def __len__(self):
        return len(self.starts)

    def batch(self, idx: torch.Tensor):
        idx = idx.to(self.device)
        g = self.starts[idx][:, None] + self._ar
        rcg = patchify(self.rcg[g])
        ecg = minmax(self.ecg[g])
        hr = self.hr[idx] if self.hr is not None else None
        return rcg, ecg, self.pos[self.tidx[idx]], hr


def patchify(x: torch.Tensor) -> torch.Tensor:
    """(B, 512, 50) → (B, 16, 50, 32)."""
    B = x.shape[0]
    return x.view(B, N_PATCH, PATCH, -1).permute(0, 1, 3, 2).contiguous()


def minmax(e: torch.Tensor) -> torch.Tensor:
    """窓ごとに [-1, 1] へ(ECG の正規化は論文・公式に記載なし)."""
    lo, hi = e.min(1, keepdim=True).values, e.max(1, keepdim=True).values
    return (e - lo) / (hi - lo).clamp_min(1e-6) * 2 - 1
