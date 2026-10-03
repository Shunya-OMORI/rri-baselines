"""radarODE-MTL: 彼らのリポジトリのルール(公開分割)7.20 ms と我々ルール(LOSO)18.48 ms の差を, 条件を 1 つずつ入れ替えて分解する(学習なし).
  .venv/bin/python scripts/radarode_gap_decomp.py
使うもの: runs/radarode_ew_wsst_official200/(公開分割, 予測は test 最小の epoch 17, metrics.json に全エポックの test 誤差)
          runs/radarode_ew_wsst200/fold*/predictions.npz(LOSO, val 最良の重みで test 参加者を予測)
読み出しはどちらも損失と整合する argmax+1. 正解は窓中央をまたぐ拍(提案手法と同じ).
注意: LOSO の metrics.json の test 欄は val 参加者の値(学習中の評価に val を渡していた)なので, test 参加者のエポックごとの誤差は無い."""
import json
from pathlib import Path

import numpy as np

from rri_baselines.protocol import eval_windows, load_trial, loso_folds, trials_by_subject, trials_of

REPO = Path(__file__).resolve().parents[1]
OFF = REPO / "runs/radarode_ew_wsst_official200"
LOSO = REPO / "runs/radarode_ew_wsst200"
PUB_TEST = list(range(75, 86))

off_pred = {int(k[5:-8]): v for k, v in np.load(OFF / "predictions.npz").items() if k.endswith("_pred_ms")}
loso_pred, val_best = {}, {}
for fold in loso_folds():
    d = LOSO / f"fold{fold.index:02d}_test{fold.test[0]}"
    z = np.load(d / "predictions.npz")
    loso_pred.update({t: z[f"trial{t}_pred_ms"] for t in trials_of(fold.test)})
    val_best[fold.test[0]] = json.loads((d / "best_epoch.json").read_text())["epoch"]
bys = trials_by_subject()
subj_of = {t: s for s, ts in bys.items() for t in ts}
truth = {}


def errs(t: int, pred: np.ndarray, sub: int) -> np.ndarray:
    if t not in truth:
        tr = load_trial(t)
        truth[t] = eval_windows(len(tr["rcg_z"]), tr["peaks"])[1]
    y, p = truth[t][::sub], pred[::sub]
    ok = np.isfinite(y)
    return np.abs(p[ok] - y[ok])


def pooled(preds: dict, trials: list, sub: int = 4) -> float:          # 窓をまとめて平均(公開コードの集計)
    return float(np.mean(np.concatenate([errs(t, preds[t], sub) for t in trials])))


def trial_subj(preds: dict, trials: list, sub: int) -> float:          # 試行 → 参加者 → 参加者間の平均(我々の採点)
    tm = {t: float(np.mean(errs(t, preds[t], sub))) for t in trials}
    ss = {}
    for t, v in tm.items():
        ss.setdefault(subj_of[t], []).append(v)
    return float(np.mean([np.mean(v) for v in ss.values()]))


hist = json.loads((OFF / "metrics.json").read_text())["history"]
test_curve = np.array([h["test"]["metrics"]["PPI"][0] * 1000 for h in hist])  # 公開分割の test(= val)を全エポック

all_trials = sorted(loso_pred)
s2930 = sorted(t for t in all_trials if subj_of[t] in (29, 30))
steps = []
s0 = pooled(off_pred, PUB_TEST)
steps.append(("0 彼らのルール: 公開分割で学習, test が最小の epoch 17 を採用, 試行 75〜85 の窓をまとめて平均(0.8 s 刻み)", s0,
              f"記録の値 {test_curve[17]:.2f}(R3 の照合)"))
eps = sorted(val_best.values())
s1 = float(np.mean(test_curve[eps]))
steps.append(("1 エポックの選び方: test 最小 → LOSO の val が選んだエポック(11 分割の {} の位置)で読む".format(eps), s1,
              f"範囲 {test_curve[eps].min():.2f}〜{test_curve[eps].max():.2f}. 最終 epoch 199 なら {test_curve[-1]:.2f}"))
s2 = pooled(loso_pred, PUB_TEST)
steps.append(("2 被験者リーク: 同じ参加者の他の試行を学習に含む公開分割 → 参加者ごと外す LOSO(val 選択)の予測", s2,
              "学習の参加者も 10 名 → 9 名(1 名は val)に減る"))
s2930_88 = [t for t in s2930 if t <= 88]
s3 = pooled(loso_pred, s2930_88)
steps.append(("3 試行: 75〜85 → 参加者 29・30 の試行のうち公開コードが読む範囲(57〜88, {} 試行)".format(len(s2930_88)), s3, ""))
s3b = pooled(loso_pred, s2930)
steps.append(("4 試行: 公開コードが読まない 89〜91(参加者 30 の運動後 3 試行)を加える", s3b, "公開コードは試行 1〜88 しか読まない"))
s4 = pooled(loso_pred, all_trials)
steps.append(("5 参加者: 29・30 の 2 名 → 11 名全員(全 91 試行, 窓をまとめて平均)", s4, ""))
s5 = trial_subj(loso_pred, all_trials, 4)
steps.append(("6 集計: 窓をまとめて平均 → 試行 → 参加者 → 11 名平均", s5, ""))
s6 = trial_subj(loso_pred, all_trials, 1)
steps.append(("7 刻み: 0.8 s → 0.2 s(全窓)= 我々ルール", s6, ""))
t88 = [t for t in all_trials if t <= 88]
order = [("89〜91 を加える(11 名の段で)", pooled(loso_pred, t88), pooled(loso_pred, all_trials)),
         ("89〜91 を加える(11 名, 我々の集計で)", trial_subj(loso_pred, t88, 4), trial_subj(loso_pred, all_trials, 4)),
         ("集計を変える(75〜85 の段で)", pooled(loso_pred, PUB_TEST), trial_subj(loso_pred, PUB_TEST, 4)),
         ("集計を変える(29・30 の全試行の段で)", pooled(loso_pred, s2930), trial_subj(loso_pred, s2930, 4)),
         ("集計を変える(11 名・試行 1〜88 の段で)", pooled(loso_pred, t88), trial_subj(loso_pred, t88, 4))]
lines = ["# radarODE-MTL: 彼らのルール 7.20 ms と我々ルール 18.48 ms の差の分解(学習なし)", "",
         "条件を上から 1 つずつ入れ替えた. 差は入れ替える順序に依存する(下の注を参照).", "",
         "| 段 | MAE [ms] | 前の段との差 | 備考 |", "|---|---|---|---|"]
prev = None
for name, v, note in steps:
    lines.append(f"| {name} | {v:.2f} | {'' if prev is None else f'{v - prev:+.2f}'} | {note} |")
    prev = v
lines += ["", "## 順序を変えた場合(同じ入れ替えを別の段で行う)", "", "| 入れ替え | 前 | 後 | 差 |", "|---|---|---|---|"]
lines += [f"| {n} | {a:.2f} | {b:.2f} | {b - a:+.2f} |" for n, a, b in order]
per = {s: trial_subj(loso_pred, [t for t in all_trials if subj_of[t] == s], 1) for s in sorted(bys)}
lines += ["", "参加者別(我々ルール, 0.2 s): " + ", ".join(f"{s}: {v:.1f}" for s, v in per.items()),
          f"LOSO の val 最良エポック: {val_best}"]
out = REPO / "results/radarode_gap_decomp.md"
out.write_text("\n".join(lines) + "\n")
print("\n".join(lines))
