"""正解を変えて学習した提案手法(runs_label_mid_linear)と既存(runs)の予測を, 両方の正解で採点して突き合わせる.
集計は公開実装と同じ(試行 MAE → 参加者平均 → 11 名平均).
  (cd mmwave-rri && .venv/bin/python ../rri-baselines/scripts/label_model_compare.py)"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/shunya/research/mmwave-rri/src")
from mmwave_rri.data_setup.loso_splits import all_subject_ids, make_loso_splits, trial_ids_for_subjects  # noqa: E402
from mmwave_rri.data_setup.paths import trial_npz  # noqa: E402
from mmwave_rri.data_setup.windows import rr_mid_linear_time, rr_spanning_time  # noqa: E402

MM = Path("/home/shunya/research/mmwave-rri")
OUT = Path("/home/shunya/research/rri-baselines/results/label_model_compare.md")
FS, WIN, STEP = 200, 800, 40
MODELS = {"既存(正解 = 窓中央をまたぐ RR)": MM / "runs", "新(正解 = 中点タコグラムの線形補間)": MM / "runs_label_mid_linear"}
LABELS = {"spanning": rr_spanning_time, "mid_linear": rr_mid_linear_time}


def main() -> None:
    table = {m: {lab: {"grid": {}, "refined": {}} for lab in LABELS} for m in MODELS}
    for fold in make_loso_splits(all_subject_ids()):
        for m, root in MODELS.items():
            pz = np.load(root / fold.run_dir.name / "predictions.npz")
            for t in trial_ids_for_subjects(fold.test):
                z = np.load(trial_npz(t))
                n, pred = len(z["rcg_z"]), pz[f"trial{t}_pred_ms"]
                starts = np.arange(0, n - WIN + 1, STEP)
                for lab, fn in LABELS.items():
                    for kind, pk in (("grid", z["peaks"]), ("refined", z["peaks_refined"])):
                        y = np.array([fn(pk, s + WIN / 2.0, s, s + WIN, FS) or np.nan for s in starts], dtype=float) * 1000
                        ok = np.isfinite(y)
                        table[m][lab][kind].setdefault(fold.test[0], []).append(float(np.mean(np.abs(pred[ok] - y[ok]))))
    lines = ["# 正解の作り方 × 学習した正解: 提案手法の 11 名平均 MAE [ms](中央値 3 後, LOSO 11 分割)", "",
             "| 学習した正解 \\ 採点の正解 | またぐ RR(5 ms 格子) | またぐ RR(放物線補間) | 中点線形補間(5 ms 格子) | 中点線形補間(放物線補間) |", "|---|---|---|---|---|"]
    for m in MODELS:
        cells = []
        for lab in LABELS:
            for kind in ("grid", "refined"):
                cells.append(f"{np.mean([np.mean(v) for v in table[m][lab][kind].values()]):.3f}")
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    lines += ["", "参加者別(採点 = 学習と同じ正解, 5 ms 格子):", "", "| 参加者 | " + " | ".join(MODELS) + " |", "|---|---|---|"]
    for s in table[list(MODELS)[0]]["spanning"]["grid"]:
        a = np.mean(table[list(MODELS)[0]]["spanning"]["grid"][s]); b = np.mean(table[list(MODELS)[1]]["mid_linear"]["grid"][s])
        lines.append(f"| {s} | {a:.2f} | {b:.2f} |")
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
