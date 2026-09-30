"""再構成した ECG の 4 s 窓から, 窓中央をまたぐ 1 拍の RR[ms] を取る(ECG を出す手法すべてに同じ規則を当てる).

規則(2026-09-30 ユーザ決定. MMECG 原著 p.280 と同じく NeuroKit2 で R 波を取る):
  1. 窓の中の波形だけに, 正解ラベルと同じ版・同じ手順(nk.ecg_clean → nk.ecg_peaks, neurokit2 0.2.13)をかける
  2. 窓中央をまたぐ 2 つの検出ピークの間隔を予測とする
  3. またぐ組が無ければ窓の中の RR の中央値, それも無ければ学習側の平均 RR(除外なしで採点するため必ず値を出す)

python -m rri_baselines.ecg_rri floor  … 正解の ECG を通したときの誤差(完全な再構成でも残る床)
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import neurokit2 as nk
import numpy as np

from .protocol import FS, WINDOW_LEN, eval_windows, load_trial, loso_folds, score, trials_of

FALLBACK_CENTER, FALLBACK_MEDIAN, FALLBACK_MEAN = 0, 1, 2


def detect_peaks(ecg: np.ndarray, fs: int = FS) -> np.ndarray:
    """R 波の位置[標本] (n,). mmwave-rri の detect_r_peaks(正解ラベル)と同じ手順."""
    try:
        cleaned = nk.ecg_clean(np.asarray(ecg, dtype=np.float64), sampling_rate=fs)
        _, info = nk.ecg_peaks(cleaned, sampling_rate=fs)
        return np.asarray(info["ECG_R_Peaks"], dtype=np.int64)
    except Exception:       # 平らな波形などで NeuroKit2 が落ちる窓は「検出なし」として扱う
        return np.zeros(0, dtype=np.int64)


def window_rr(ecg_window: np.ndarray, fallback_ms: float, fs: int = FS) -> tuple[float, int]:
    """窓 (WINDOW_LEN,) の波形から (RR[ms], どの規則で出したか)."""
    peaks = detect_peaks(ecg_window, fs)
    k = int(np.searchsorted(peaks, len(ecg_window) / 2.0))
    if 0 < k < len(peaks):
        return float(peaks[k] - peaks[k - 1]) / fs * 1000.0, FALLBACK_CENTER
    if len(peaks) >= 2:
        return float(np.median(np.diff(peaks))) / fs * 1000.0, FALLBACK_MEDIAN
    return float(fallback_ms), FALLBACK_MEAN


def trial_predictions(ecg: np.ndarray, n_samples: int, peaks_true: np.ndarray, fallback_ms: float):
    """試行の ECG(再構成でも正解でもよい, 試行と同じ長さ)から, eval_windows と同じ並びの予測と規則の内訳."""
    starts, _ = eval_windows(n_samples, peaks_true)
    out = [window_rr(ecg[s:s + WINDOW_LEN], fallback_ms) for s in starts]
    return np.array([v for v, _ in out]), np.array([r for _, r in out])


def train_mean_rr(train_trials: list[int]) -> float:
    rr = []
    for t in train_trials:
        trial = load_trial(t)
        _, true_ms = eval_windows(len(trial["rcg_z"]), trial["peaks"])
        rr.append(true_ms[np.isfinite(true_ms)])
    return float(np.mean(np.concatenate(rr)))


def _floor_one(args):
    t, fallback_ms = args
    trial = load_trial(t)
    pred, rule = trial_predictions(trial["ecg"], len(trial["rcg_z"]), trial["peaks"], fallback_ms)
    return t, pred, rule


def floor(workers: int = 2) -> dict:
    jobs = []
    for fold in loso_folds():
        fallback_ms = train_mean_rr(trials_of(fold.train))
        jobs += [(t, fallback_ms) for t in trials_of(fold.test)]
    with Pool(workers) as pool:
        done = pool.map(_floor_one, jobs)
    preds = {t: p for t, p, _ in done}
    rules = np.concatenate([r for _, _, r in done])
    result = score(preds)
    result["rule_share"] = {name: float(np.mean(rules == code)) for name, code in
                            (("center", FALLBACK_CENTER), ("median", FALLBACK_MEDIAN), ("train_mean", FALLBACK_MEAN))}
    return result


if __name__ == "__main__":
    if sys.argv[1:2] == ["floor"]:
        res = floor(int(sys.argv[2]) if len(sys.argv) > 2 else 2)
        out = Path("results/ecg_route_floor.json")
        out.write_text(json.dumps(res, indent=1, ensure_ascii=False))
        print(f"正解 ECG を通した床: {res['mae_ms']:.3f} ms  規則の内訳 {res['rule_share']}")
