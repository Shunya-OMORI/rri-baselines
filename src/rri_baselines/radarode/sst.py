"""radarODE-MTL の入力(SST 画像)を, 原著の MMECG_to_SST.m と同じ順序で試行全体について作る.

MMECG_to_SST.m(radarODE-MTL リポジトリ): 点ごとに試行全体へ wsst(10 voices/octave) → 振幅の高周波側の半分(71 行)
→ 時間を 200 → 30 Hz に resample → 点ごとに試行全体で min-max 正規化 → 学習時に 4 s(120 フレーム)を切り出す.
MATLAB の wsst は ssqueezepy の ssq_cwt(morlet, nv=10, 対数の尺度)で代用する(同一ではない. README に明記).
周波数の行は低い順に並べる(著者の配布データ data_RFcardi の SST と突き合わせて, 高周波側の半分を昇順に並べた形が最も近かった).

  .venv/bin/python -m rri_baselines.radarode.sst [並列数]     → cache/radarode_sst/trial<id>.npy (50, 71, T30) float16
"""
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from rri_baselines.protocol import load_trial, trials_by_subject

REPO = Path(__file__).resolve().parents[3]
CACHE = Path(os.environ.get("RRI_SST_CACHE", REPO / "cache/radarode_sst"))
N_ROWS, F_OUT = 71, 30


def trial_sst(rcg: np.ndarray) -> np.ndarray:
    import ssqueezepy as ssq
    from scipy.signal import resample_poly
    out = []
    for c in range(rcg.shape[1]):
        tx, _, freqs, _ = ssq.ssq_cwt(rcg[:, c].astype(np.float64), wavelet="morlet", fs=200, nv=10, scales="log")
        order = np.argsort(np.asarray(freqs))[-N_ROWS:]              # 高い方から 71 行を, 低い順に
        mag = np.abs(np.asarray(tx))[order]
        out.append(resample_poly(mag, 3, 20, axis=1))                # 200 → 30 Hz(MATLAB の resample に相当)
    sst = np.stack(out)
    lo = sst.min(axis=(1, 2), keepdims=True)
    hi = sst.max(axis=(1, 2), keepdims=True)
    return ((sst - lo) / (hi - lo + 1e-12)).astype(np.float16)


def compute(trial_id: int) -> str:
    path = CACHE / f"trial{trial_id}.npy"
    if path.exists():
        return f"trial {trial_id}: skip"
    CACHE.mkdir(parents=True, exist_ok=True)
    sst = trial_sst(load_trial(trial_id)["rcg_z"])
    tmp = path.with_suffix(".tmp.npy")
    np.save(tmp, sst)
    tmp.rename(path)
    return f"trial {trial_id}: {sst.shape}"


if __name__ == "__main__":
    CACHE.mkdir(parents=True, exist_ok=True)
    ids = [t for ts in trials_by_subject().values() for t in ts]
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 4) as pool:
        for line in pool.imap_unordered(compute, ids):
            print(line, flush=True)
