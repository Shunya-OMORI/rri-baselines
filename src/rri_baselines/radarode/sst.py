"""radarODE-MTL の入力(SST 画像)を作る. 試行全体に 1 回だけ変換をかけ, 学習・評価で 4 s(120 フレーム)を切り出す.

原著の MMECG_to_SST.m は「試行全体に wsst(10 voices/octave) → 振幅の行の後半(71 行) → 30 Hz → 点ごとに試行全体で min-max」.
MATLAB が手元に無いので wsst は ssqueezepy の ssq_cwt(morlet)で代用する. このとき行の並び(昇順か降順か)が分からず,
「後半」が 0.85〜100 Hz か低い側かが決まらない. 0.85〜100 Hz・試行全体の正規化で学習すると val の PPI 誤差が 115 ms で,
定数の予測(約 90 ms)より悪かった(2026-09-28). そこで, 別セッションの再現(data_prepared_cft, val 約 9 ms)と同じ
「0.2〜7.7 Hz の帯域(窓単位の変換の後半の行, 高い方から並べる)を 71 行に補間・窓ごと点ごとに min-max」に合わせる(相手に有利な側).
窓ごとの正規化は data.py で行う. ここでは試行ごとに最大値で割った振幅を float16 で保存する.

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


F_LO, F_HI = 0.2, 7.72          # data_prepared_cft の帯域(800 点の窓の ssq_cwt の後半の行)


def trial_sst(rcg: np.ndarray) -> np.ndarray:
    import ssqueezepy as ssq
    from scipy.signal import resample_poly
    out = []
    for c in range(rcg.shape[1]):
        tx, _, freqs, _ = ssq.ssq_cwt(rcg[:, c].astype(np.float64), wavelet="morlet", fs=200, nv=10, scales="log")
        freqs = np.asarray(freqs)
        rows = np.flatnonzero((freqs >= F_LO) & (freqs <= F_HI))
        rows = rows[np.argsort(freqs[rows])[::-1]]                     # 高い方から(data_prepared_cft と同じ並び)
        mag = resample_poly(np.abs(np.asarray(tx))[rows], 3, 20, axis=1)          # 200 → 30 Hz
        interp = np.stack([np.interp(np.linspace(0, 1, N_ROWS), np.linspace(0, 1, len(rows)), e) for e in np.eye(len(rows))], 1)
        out.append(interp @ mag)                                        # 行方向の線形補間(71 行)
    sst = np.stack(out)
    return (sst / (sst.max(axis=(1, 2), keepdims=True) + 1e-12)).astype(np.float16)


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
