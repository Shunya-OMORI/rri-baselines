"""MATLAB の wsst(解析 Morlet 'amor', VoicesPerOctave = 10)を移した SST. 原著 MMECG_to_SST.m と著者のデータ生成(CFT-RFcardi の
utils/data_generation.ipynb, MMECG 用のセル)を順にたどる.

  1. 試行全体(35,505 点)に wsst. 行は fs/N 〜 fs/2 を対数等間隔に 140 本(オクターブ数 floor(log2 N) − 1 = 14, × 10 本)
  2. 振幅の行の後半 71 本(MATLAB の 70〜140 行 = 約 0.72〜100 Hz. 低い方から並ぶ)
  3. 時間方向を 200 → 30 Hz(MATLAB の resample の代わりに resample_poly)
  4. 点ごとに試行全体で min-max → [0, 1]
  5. 行を上下反転(load_sst の np.flip. 行 0 が 100 Hz)
窓ごとの min-max(sst_norm_01)は切り出しの後に data 側で行う.

  python -m rri_baselines.radarode.wsst [cuda|cpu]   → cache/radarode_wsst/trial<id>.npy (50, 71, T30) float32
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch

from rri_baselines.protocol import load_trial, trials_by_subject

REPO = Path(__file__).resolve().parents[3]
CACHE = Path(os.environ.get("RRI_WSST_CACHE", REPO / "cache/radarode_wsst"))
NV, FS, F_OUT = 10, 200, 30


def wsst(x: torch.Tensor, fs: float = FS, nv: int = NV) -> tuple[torch.Tensor, np.ndarray]:
    """x (N,) 実数 → SST (na, N) 複素, 行の周波数 [Hz](低い方から). MATLAB wsst の既定(amor, 埋めなし, しきい値 1e-8)."""
    n = x.numel()
    na = (int(np.floor(np.log2(n))) - 1) * nv
    x = x.double() - x.double().mean()
    scales = (2.0 ** (torch.arange(1, na + 1, dtype=torch.float64, device=x.device) / nv))[:, None]
    omega = 2 * np.pi * torch.fft.fftfreq(n, dtype=torch.float64, device=x.device)[None]        # rad/標本
    pos = omega > 0
    psi = 2.0 * torch.exp(-0.5 * (6.0 - scales * omega) ** 2) * pos                               # amor: 中心 6 rad
    xdft = torch.fft.fft(x)[None]
    cwt = torch.fft.ifft(xdft * psi, dim=1)
    dcwt = torch.fft.ifft(xdft * psi * (1j * omega), dim=1)
    inst = (dcwt / cwt).imag / (2 * np.pi) * fs                                                    # 瞬時周波数 [Hz]
    inst = torch.where(inst.abs() < 1e-8, torch.full_like(inst, float("nan")), inst.abs())
    lo, hi = np.log2(fs / n), np.log2(fs / 2)
    row = torch.floor(na / (hi - lo) * (torch.log2(inst) - lo))                                   # 0 始まりの行
    ok = torch.isfinite(row) & (row >= 0) & (row < na)
    flat = (row.clamp(0, na - 1).long() * n + torch.arange(n, device=x.device)[None])[ok]
    out = torch.zeros(na * n, dtype=torch.complex128, device=x.device)
    out.index_add_(0, flat, cwt[ok])
    return out.view(na, n) * (np.log(2) / nv), 2.0 ** np.linspace(lo, hi, na)


def trial_sst(rcg: np.ndarray, device: str) -> np.ndarray:
    from scipy.signal import resample_poly
    out = []
    for c in range(rcg.shape[1]):
        sst, _ = wsst(torch.from_numpy(rcg[:, c].astype(np.float64)).to(device))
        half = sst[sst.shape[0] // 2 - 1:].abs().cpu().numpy()                                     # MATLAB の freq_len/2:freq_len(71 行)
        half = resample_poly(half, 3, 20, axis=1)                                                  # 200 → 30 Hz
        half = (half - half.min()) / (half.max() - half.min())                                     # 点ごとに試行全体で [0, 1]
        out.append(half[::-1])                                                                     # 上下反転(行 0 が 100 Hz)
    return np.stack(out).astype(np.float32)


if __name__ == "__main__":
    device = sys.argv[1] if len(sys.argv) > 1 else ("cuda" if torch.cuda.is_available() else "cpu")
    CACHE.mkdir(parents=True, exist_ok=True)
    for t in [t for ts in trials_by_subject().values() for t in ts]:
        path = CACHE / f"trial{t}.npy"
        if path.exists():
            continue
        sst = trial_sst(load_trial(t)["rcg_z"], device)
        tmp = path.with_suffix(".tmp.npy")
        np.save(tmp, sst)
        tmp.rename(path)
        print(f"trial {t}: {sst.shape}", flush=True)
