"""各手法の予測を同じ採点で集計し, 比較表 results/summary.md を作る.

行: 提案手法(mmwave-rri の 91131 相当)・先行手法(runs/<手法>/fold*/predictions.npz)・定数(入力を見ず学習側の平均 RR を出す, R-E4)
列: 参加者平均の MAE と, 11 名それぞれの MAE. 採点は protocol.score(除外なし・試行平均 → 参加者平均).

  .venv/bin/python -m rri_baselines.report
"""
import json
from pathlib import Path

import numpy as np

from rri_baselines.protocol import FS, eval_windows, load_trial, loso_folds, score, trials_of

REPO = Path(__file__).resolve().parents[2]
MMWAVE_RUNS = Path("/home/shunya/research/mmwave-rri/runs")
METHODS = {"wu2019": ("Wu・Sakamoto 2019", "trial{t}_pred_ms"),
           "mibinet": ("MIBINET 2023", "trial{t}_pred_ms"),
           "radarode_ew": ("radarODE-MTL(等重み, 20 エポック)", "trial{t}_pred_ms"),
           "radarode_ew200": ("radarODE-MTL(等重み, 200 エポック)", "trial{t}_pred_ms")}
# 原論文の報告値. データ・分割・目標量・指標がそれぞれ違うので, 上の表と同じ列には入れない(Fable の助言どおり併記だけ)
PAPER = [("EQ-Radio 2016", "IBI 誤差の平均 3.2 ms(p.2・p.8)", "自前データ. 生のレーダから位相を取る前段込み. 拍ごとの IBI"),
         ("Wu・Sakamoto 2019", "RMSE 平均 20.6 ms, 絶対誤差 15.6 ± 13.5 ms", "自前 7 名, 被験者ごとの学習. 入力は呼吸を含む変位"),
         ("MIBINET 2023", "RMSE 20.69 ms(Table 3・4, Radar-HS)", "推論時に心電図の R 波で窓を切り, 7 窓平均と平滑化の後. 連結 RMSE"),
         ("radarODE-MTL(等重み)", "PPI 誤差 8.0 ms(Table I, p.8)", "MMECG 11 分割(被験者単位), 4 s・ストライド 1 s の窓, 200 エポック, 5 回平均. "
                                                                       "単一タスク 9.6 ms, EGA(T = 1.0)8.5 ms")]


def collect(run_root: Path, key: str) -> dict | None:
    preds = {}
    for fold in loso_folds():
        f = run_root / f"fold{fold.index:02d}_test{fold.test[0]}/predictions.npz"
        if not f.exists():
            return None
        z = np.load(f)
        preds.update({t: z[key.format(t=t)] for t in trials_of(fold.test)})
    return preds


def constant_predictions() -> dict:
    preds = {}
    for fold in loso_folds():
        rr = np.concatenate([np.diff(load_trial(t)["peaks"]) for t in trials_of(fold.train)]) / FS * 1000.0
        for t in trials_of(fold.test):
            n = len(load_trial(t)["rcg_z"])
            preds[t] = np.full(len(eval_windows(n, np.array([0.0]))[0]), rr.mean())
    return preds


def main() -> None:
    rows = {"提案手法(公開実装 91131 相当)": collect(MMWAVE_RUNS, "trial{t}_pred_ms")}
    eq = json.loads((REPO / "results/eqradio/evaluation.json").read_text())
    for name, (label, key) in METHODS.items():
        rows[label] = collect(REPO / "runs" / name, key)
    rows["定数(学習側の平均 RR)"] = constant_predictions()
    results = {label: score(p) for label, p in rows.items() if p is not None}
    results["EQ-Radio 2016(学習なし)"] = {"mae_ms": eq["mae_ms"], "subject_mae_ms": {int(k): v for k, v in eq["subject_mae_ms"].items()}}
    subjects = sorted(next(iter(results.values()))["subject_mae_ms"])
    lines = ["| 手法 | 参加者平均 MAE [ms] | " + " | ".join(f"s{s}" for s in subjects) + " |",
             "|---|---|" + "---|" * len(subjects)]
    for label, r in sorted(results.items(), key=lambda kv: kv[1]["mae_ms"]):
        sm = {int(k): v for k, v in r["subject_mae_ms"].items()}
        lines.append(f"| {label} | {r['mae_ms']:.2f} | " + " | ".join(f"{sm[s]:.1f}" for s in subjects) + " |")
    missing = [label for label, p in rows.items() if p is None]
    text = "\n".join(lines) + ("\n\n未完了: " + "・".join(missing) if missing else "") + "\n"
    text += "\n原論文の報告値(条件が違うので上の表とは比べない)\n\n| 手法 | 報告値 | 条件 |\n|---|---|---|\n"
    text += "".join(f"| {m} | {v} | {c} |\n" for m, v, c in PAPER)
    (REPO / "results/summary.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
