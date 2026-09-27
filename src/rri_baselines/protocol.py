"""比較の条件: 入力・分割・目標量・採点を提案手法(公開実装 mmwave-rri)と同じにする.

入力   : 公開 MMECG の RCG 50 点(胸壁の加速度, 200 Hz). 前処理済みの試行は mmwave-rri の data/processed を読む
分割   : 11 名を 1 名ずつ test に残す LOSO. fold i は test が i 番目, val が (i+1) 番目の参加者
目標量 : 4 s 窓を 0.2 s ずつ進め, 窓の中央をまたぐ 1 拍の RR[ms]. 両端の R 波が窓の中にある窓だけ正解を持つ
採点   : 正解を持つ窓すべての絶対誤差(除外なし) → 試行の平均 → 参加者の平均 → 11 名の平均
"""
import csv
import os
from pathlib import Path
from typing import NamedTuple

import numpy as np

FS = 200
WINDOW_SEC = 4.0
STEP_SEC = 0.2
WINDOW_LEN = int(WINDOW_SEC * FS)       # 800 標本
STEP = int(round(STEP_SEC * FS))        # 40 標本
DATA_ROOT = Path(os.environ.get("RRI_DATA_ROOT", "/home/shunya/research/mmwave-rri/data/processed"))


class Fold(NamedTuple):
    index: int
    train: list[int]
    val: list[int]
    test: list[int]


def trials_by_subject(root: Path = DATA_ROOT) -> dict[int, list[int]]:
    mapping: dict[int, list[int]] = {}
    with (root / "trials.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            mapping.setdefault(int(row["subject_id"]), []).append(int(row["trial_id"]))
    return {s: sorted(ids) for s, ids in sorted(mapping.items())}


def loso_folds(root: Path = DATA_ROOT) -> list[Fold]:
    subjects = list(trials_by_subject(root))
    return [Fold(i, [s for s in subjects if s not in (t, subjects[(i + 1) % len(subjects)])],
                 [subjects[(i + 1) % len(subjects)]], [t]) for i, t in enumerate(subjects)]


def trials_of(subjects: list[int], root: Path = DATA_ROOT) -> list[int]:
    by_subject = trials_by_subject(root)
    return sorted(t for s in subjects for t in by_subject[s])


def load_trial(trial_id: int, root: Path = DATA_ROOT) -> dict:
    """rcg_z (T, 50)・ecg (T,)・peaks(5 ms 格子の R 波)・peaks_refined・subject_id など."""
    with np.load(root / f"trial{trial_id}.npz") as z:
        return {k: z[k] for k in z.files}


def rr_spanning(peaks: np.ndarray, at: float, start: int, end: int) -> float:
    """時刻 at[標本] をまたぐ RR[標本]. 両端の R 波が [start, end) の外なら NaN."""
    k = int(np.searchsorted(peaks, at))
    if k == 0 or k >= len(peaks) or peaks[k - 1] < start or peaks[k] >= end:
        return np.nan
    return float(peaks[k] - peaks[k - 1])


def eval_windows(n_samples: int, peaks: np.ndarray, step: int = STEP) -> tuple[np.ndarray, np.ndarray]:
    """評価の窓の開始位置 (n,) と, 窓中央の正解 RR[ms] (n,). 正解を作れない窓は NaN."""
    starts = np.arange(0, n_samples - WINDOW_LEN + 1, step)
    rr = np.array([rr_spanning(peaks, s + WINDOW_LEN / 2.0, s, s + WINDOW_LEN) for s in starts])
    return starts, rr / FS * 1000.0


def score(pred_ms_by_trial: dict[int, np.ndarray], root: Path = DATA_ROOT) -> dict:
    """試行ごとの窓の予測[ms](eval_windows と同じ並び)を採点する. 参加者の平均の 11 名平均を返す."""
    by_subject = trials_by_subject(root)
    trial_mae = {}
    for t, pred in pred_ms_by_trial.items():
        trial = load_trial(t, root)
        _, true_ms = eval_windows(len(trial["rcg_z"]), trial["peaks"])
        assert len(pred) == len(true_ms), (t, len(pred), len(true_ms))
        ok = np.isfinite(true_ms)
        trial_mae[t] = float(np.mean(np.abs(np.asarray(pred)[ok] - true_ms[ok])))
    subject_mae = {s: float(np.mean([trial_mae[t] for t in ids])) for s, ids in by_subject.items()
                   if all(t in trial_mae for t in ids)}
    return {"mae_ms": float(np.mean(list(subject_mae.values()))), "subject_mae_ms": subject_mae,
            "trial_mae_ms": trial_mae}
