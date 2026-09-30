"""試行の状態(MMECG の physistatus: NB 通常呼吸・IB 不規則呼吸・SP 発話・PE 運動後)ごとに各手法の誤差を集計する.

誤差は試行ごとの窓 MAE を状態ごとに試行で平均したもの(参加者平均ではない). あわせて, 予測と正解の比の分布で
「倍の周期を拾う誤り」(比 1.6〜2.4)と「全体に長め・短めへ寄る誤り」(比 1.2〜1.6, 0.6〜0.8)を分ける.

  .venv/bin/python -m rri_baselines.by_status      → results/by_status.md
"""
from pathlib import Path

import numpy as np

from rri_baselines.protocol import DATA_ROOT, eval_windows, load_trial, loso_folds, trials_of
from rri_baselines.report import MMWAVE_RUNS, REPO

METHODS = {"提案手法": MMWAVE_RUNS, "radarODE-MTL 20 ep": REPO / "runs/radarode_ew", "radarODE-MTL 200 ep": REPO / "runs/radarode_ew200",
           "MIBINET": REPO / "runs/mibinet", "Wu・Sakamoto": REPO / "runs/wu2019"}
STATUS = ["NB", "IB", "SP", "PE"]
BINS = [(0, 0.6), (0.6, 0.8), (0.8, 1.2), (1.2, 1.6), (1.6, 2.4), (2.4, np.inf)]


def main() -> None:
    rows = []
    for fold in loso_folds():
        name = f"fold{fold.index:02d}_test{fold.test[0]}/predictions.npz"
        zs = {m: np.load(r / name) for m, r in METHODS.items()}
        for t in trials_of(fold.test):
            tr = load_trial(t)
            with np.load(DATA_ROOT / f"trial{t}.npz") as z:
                status = str(z["physistatus"])
            _, true = eval_windows(len(tr["rcg_z"]), tr["peaks"])
            ok = np.isfinite(true)
            pred = {m: z[f"trial{t}_pred_ms"][ok] for m, z in zs.items()}
            rows.append({"status": status, "rr": true[ok].mean(), "true": true[ok], "pred": pred,
                         "mae": {m: np.abs(p - true[ok]).mean() for m, p in pred.items()}})
    lines = ["| 状態 | 試行数 | 正解 RR の平均 [ms] | " + " | ".join(METHODS) + " |", "|---|---|---|" + "---|" * len(METHODS)]
    for st in STATUS:
        r = [x for x in rows if x["status"] == st]
        lines.append(f"| {st} | {len(r)} | {np.mean([x['rr'] for x in r]):.0f} | "
                     + " | ".join(f"{np.mean([x['mae'][m] for x in r]):.1f}" for m in METHODS) + " |")
    lines += ["", "予測 / 正解 の比の分布(全窓, %)", "",
              "| 手法 | 状態 | " + " | ".join(f"{lo}〜{hi}" for lo, hi in BINS) + " |", "|---|---|" + "---|" * len(BINS)]
    for m in METHODS:
        for st in ("NB", "PE"):
            ratio = np.concatenate([x["pred"][m] / x["true"] for x in rows if x["status"] == st])
            lines.append(f"| {m} | {st} | " + " | ".join(f"{100 * np.mean((ratio >= lo) & (ratio < hi)):.1f}" for lo, hi in BINS) + " |")
    text = "\n".join(lines) + "\n"
    (REPO / "results/by_status.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
