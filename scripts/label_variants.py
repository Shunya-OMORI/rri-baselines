"""正解 RRI の作り方を変えたときの, 提案手法(公開実装 91131, LOSO 11 分割の predictions.npz)の誤差.
正解の候補(いずれも窓中央 t_c の値):
  spanning   : t_c をまたぐ RR(公開実装の正解. 7.019 ms の再現で較正)
  mid_linear : RR を拍の中点に置いたタコグラムの線形補間
  mid_pchip  : 同じタコグラムの単調 3 次補間(PCHIP)
  mid_cubic  : 同じタコグラムの 3 次スプライン補間
  rend_linear: RR を区間の終わりの R 波の時刻に置いたタコグラムの線形補間
  rstart_linear: RR を区間の始めの R 波の時刻に置いたタコグラムの線形補間
集計は公開実装と同じ(試行の MAE → 参加者平均 → 11 名平均). 補間できない端の窓は除く.
  (cd mmwave-rri && .venv/bin/python ../rri-baselines/scripts/label_variants.py)"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator

sys.path.insert(0, "/home/shunya/research/mmwave-rri/src")
from mmwave_rri.data_setup.loso_splits import all_subject_ids, make_loso_splits, trial_ids_for_subjects  # noqa: E402
from mmwave_rri.data_setup.paths import trial_npz  # noqa: E402
from mmwave_rri.data_setup.windows import cut_eval_windows  # noqa: E402

RUNS = Path("/home/shunya/research/mmwave-rri/runs")
OUT = Path("/home/shunya/research/rri-baselines/results/label_variants.md")
FS, WIN, STEP = 200, 800, 40
VARIANTS = ["spanning", "mid_linear", "mid_pchip", "mid_cubic", "rend_linear", "rstart_linear"]


def labels(peaks: np.ndarray, centers: np.ndarray, n: int) -> dict:
    pk = peaks.astype(float)
    rr = np.diff(pk) / FS * 1000.0
    t_mid, t_end, t_start = (pk[:-1] + pk[1:]) / 2.0, pk[1:], pk[:-1]
    _, span = cut_eval_windows(n, peaks, FS, 4.0, 0.2)
    out = {"spanning": span * 1000.0}
    for name, tt in [("mid", t_mid), ("rend", t_end), ("rstart", t_start)]:
        inside = (centers >= tt[0]) & (centers <= tt[-1])
        lin = np.where(inside, np.interp(centers, tt, rr), np.nan)
        out[f"{name}_linear"] = lin
        if name == "mid":
            out["mid_pchip"] = np.where(inside, PchipInterpolator(tt, rr)(centers), np.nan)
            out["mid_cubic"] = np.where(inside, CubicSpline(tt, rr)(centers), np.nan)
    return out


def main() -> None:
    per_subject = {v: {} for v in VARIANTS}
    per_trial = {v: {} for v in VARIANTS}
    n_windows = {v: 0 for v in VARIANTS}
    for fold in make_loso_splits(all_subject_ids()):
        pz = np.load(RUNS / fold.run_dir.name / "predictions.npz")
        subj = fold.test[0]
        for t in trial_ids_for_subjects(fold.test):
            z = np.load(trial_npz(t))
            n, peaks = len(z["rcg_z"]), z["peaks"]
            pred = pz[f"trial{t}_pred_ms"]
            starts = np.arange(0, n - WIN + 1, STEP)
            assert len(starts) == len(pred), (t, len(starts), len(pred))
            centers = starts + WIN / 2.0
            lab = labels(peaks, centers, n)
            for v in VARIANTS:
                ok = np.isfinite(lab[v])
                per_trial[v][t] = float(np.mean(np.abs(pred[ok] - lab[v][ok])))
                n_windows[v] += int(ok.sum())
        for v in VARIANTS:
            per_subject[v][subj] = float(np.mean([per_trial[v][t] for t in trial_ids_for_subjects(fold.test)]))
    lines = ["# 正解 RRI の作り方を変えたときの提案手法の誤差(公開実装 91131, LOSO 11 分割, 中央値 3 後)", "",
             "| 正解の作り方 | 11 名平均 MAE [ms] | 全窓平均 MAE [ms] | 窓の数 |", "|---|---|---|---|"]
    desc = {"spanning": "窓中央をまたぐ RR(公開実装の正解)", "mid_linear": "拍の中点に置いた RR を線形補間",
            "mid_pchip": "拍の中点に置いた RR を PCHIP(単調 3 次)補間", "mid_cubic": "拍の中点に置いた RR を 3 次スプライン補間",
            "rend_linear": "区間の終わりの R 波に置いた RR を線形補間", "rstart_linear": "区間の始めの R 波に置いた RR を線形補間"}
    for v in VARIANTS:
        allw = np.mean([per_trial[v][t] for t in per_trial[v]])   # 試行の単純平均(試行の窓数は 868 でほぼ同じなので全窓平均に近い)
        lines.append(f"| {desc[v]} | {np.mean(list(per_subject[v].values())):.3f} | {allw:.3f} | {n_windows[v]:,} |")
    lines += ["", "参加者別(11 名平均の内訳):", "", "| 参加者 | " + " | ".join(VARIANTS) + " |", "|---|" + "---|" * len(VARIANTS)]
    for s in per_subject["spanning"]:
        lines.append(f"| {s} | " + " | ".join(f"{per_subject[v][s]:.2f}" for v in VARIANTS) + " |")
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:4 + len(VARIANTS)]))
    (OUT.with_suffix(".json")).write_text(json.dumps({"per_subject": per_subject, "per_trial": per_trial}, indent=1))


if __name__ == "__main__":
    main()
