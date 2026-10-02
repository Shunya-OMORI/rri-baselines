"""我々ルール 200 エポック(radarode_ew_wsst200)の予測を採点する. 暫定(predictions_provisional.npz)か確定(predictions.npz)を引数で選ぶ.
  .venv/bin/python scripts/radarode_loso200_score.py provisional|final"""
import json
import sys
from pathlib import Path

import numpy as np

from rri_baselines.protocol import FS, WINDOW_LEN, eval_windows, load_trial, loso_folds, rr_spanning, score, trials_of

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "runs/radarode_ew_wsst200"
kind = sys.argv[1] if len(sys.argv) > 1 else "provisional"
fname = "predictions_provisional.npz" if kind == "provisional" else "predictions.npz"
preds, epochs = {}, {}
for fold in loso_folds():
    f = RUNS / f"fold{fold.index:02d}_test{fold.test[0]}" / fname
    if not f.exists():
        print("欠け:", f.parent.name); continue
    z = np.load(f)
    epochs[fold.test[0]] = int(z["best_epoch"]) if "best_epoch" in z.files else None
    preds.update({t: z[f"trial{t}_pred_ms"] for t in trials_of(fold.test)})
r_loss = score(preds)                                             # 損失整合の読み出し(argmax+1), 0.2 s 刻み全窓
r_pub = score({t: v - 1000.0 / FS * 2 for t, v in preds.items()})  # 公開式(argmax を L+1 と比較 = 読み出しが 2 標本短い)
def score_sub(sub: int) -> float:                                   # 0.8 s 刻み(4 窓に 1 つ)の全窓平均 → 試行 → 参加者 → 11 名平均
    from rri_baselines.protocol import trials_by_subject
    bys = trials_by_subject()
    tm = {}
    for t, pred in preds.items():
        tr = load_trial(t)
        _, true_ms = eval_windows(len(tr["rcg_z"]), tr["peaks"])
        p, y = pred[::sub], true_ms[::sub]
        ok = np.isfinite(y)
        tm[t] = float(np.mean(np.abs(p[ok] - y[ok])))
    subj = {s: np.mean([tm[t] for t in ts if t in tm]) for s, ts in bys.items() if any(t in tm for t in ts)}
    return float(np.mean(list(subj.values())))
r_08 = {"mae_ms": score_sub(4)}
lines = [f"# radarODE-MTL 我々ルール(LOSO 11 分割, val 選択, 200 エポック・周期 100) {'暫定' if kind == 'provisional' else '確定'}値",
         "", f"val 最良エポック: {epochs}", "",
         "| 読み出し | 11 名平均 MAE [ms] |", "|---|---|",
         f"| 損失整合(argmax+1), 0.2 s 刻み全窓 | **{r_loss['mae_ms']:.2f}** |",
         f"| 公開コードの式(2 標本ずれ), 0.2 s 刻み全窓 | {r_pub['mae_ms']:.2f} |"]
if r_08 is not None:
    lines.append(f"| 損失整合, 0.8 s 刻み(4 窓に 1 つ) | {r_08['mae_ms']:.2f} |")
lines += ["", "参加者別(損失整合, 0.2 s):", "", "| 参加者 | " + " | ".join(str(s) for s in r_loss["subject_mae_ms"]) + " |",
          "|---|" + "---|" * len(r_loss["subject_mae_ms"]),
          "| MAE | " + " | ".join(f"{v:.1f}" for v in r_loss["subject_mae_ms"].values()) + " |"]
out = REPO / f"results/radarode_loso200_{kind}.md"
out.write_text("\n".join(lines) + "\n")
print("\n".join(lines))
