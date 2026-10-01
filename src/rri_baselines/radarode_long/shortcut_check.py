"""長期モジュールが「レーダ」と「形態参照」のどちらで ECG を出しているかを, 入力を他の窓と入れ替えたときの損失で確かめる.

MMECG の再実装では, teacher forcing で TCN が正解の過去の ECG だけを見てレーダを無視した(mmecg/cond_screen.py).
radarODE の長期モジュールは正解の ECG を入力に取らない(形態参照は SCEG の出力を KDE の PPI で並べたもの)ので同じ近道は起きない.
ただし SCEG を同じ学習側の参加者で学習してから学習側の形態参照を作ると, 学習側の参照は test より正確になり(SCEG の過学習),
長期モジュールが参照だけを写す近道を覚えるおそれがある. その程度を, 次の 2 種類の参照で短く学習して測る(本学習ではない).
  template: 入力を見ずに平均形状を出すだけの SCEG(学習側の周期の平均)を KDE の PPI・位相で並べたもの(SCEG の下限)
  oracle  : 正解の ECG の周期をそのまま正しい位置に置いたもの(過学習した SCEG の極限. 方法には無い)
val の窓で (1) そのまま (2) レーダだけ他の窓と入れ替え (3) 参照だけ入れ替え (4) 両方入れ替え の RMSE を出す.
(2) で損失がほとんど増えなければレーダを使っていない. train.py の評価でも同じ swap_losses を本番の参照に当てる.

  python -m rri_baselines.radarode_long.shortcut_check <分割> <template|oracle> [歩数]   (s7 の CPU, 1 プロセス)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from rri_baselines.ecg_rri import window_rr
from rri_baselines.protocol import STEP, WINDOW_LEN, eval_windows, loso_folds, trials_of
from rri_baselines.radarode_long import ppi as kde
from rri_baselines.radarode_long.data import Trials, anchors, cycle_bounds, mechanical_delay, minmax01, place, to_len
from rri_baselines.radarode_long.model import LongTerm

REPO = Path(__file__).resolve().parents[3]
CFG = kde.parse_label("paper/lseg4/neurokit/scott")


@torch.no_grad()
def swap_losses(model, rcg: torch.Tensor, ref: torch.Tensor, y: torch.Tensor, batch: int = 64, seed: int = 0) -> dict:
    """(N, 50, 800)・(N, 1, 800)・(N, 1, 800) → 入れ替えごとの RMSE. 入れ替え先は同じ集合の別の窓(乱順)."""
    perm = torch.from_numpy(np.random.default_rng(seed).permutation(len(rcg)))
    out = {}
    for name, (a, b) in {"as_is": (rcg, ref), "swap_radar": (rcg[perm], ref), "swap_reference": (rcg, ref[perm]),
                         "swap_both": (rcg[perm], ref[perm])}.items():
        se = [F.mse_loss(model(a[i:i + batch], b[i:i + batch]), y[i:i + batch], reduction="sum").item()
              for i in range(0, len(a), batch)]
        out[name] = float(np.sqrt(np.sum(se) / y.numel()))
    out["reference_itself"] = float(torch.sqrt(F.mse_loss(ref, y)).item())
    out["constant_mean"] = float(torch.sqrt(F.mse_loss(y.mean(0, keepdim=True).expand_as(y), y)).item())
    return out


def build(trial_ids, kind: str, template: np.ndarray, delta: float, fallback: float, stride: int):
    tr = Trials(trial_ids)
    xs, refs, ys, truth = [], [], [], []
    for t in trial_ids:
        rcg, ecg, peaks = tr[t]
        z = kde.load_cache(t)
        _, true_ms = eval_windows(rcg.shape[1], peaks)
        for s in np.arange(0, rcg.shape[1] - WINDOW_LEN + 1, stride):
            j = s // STEP
            p, ph = float(z["ppi_ms"][(j,) + CFG]), float(z["phase"][(j,) + CFG])
            p = p if np.isfinite(p) else fallback
            if kind == "template":
                r = anchors(p, ph, delta)
                ref = place(torch.from_numpy(np.tile(template, (len(r), 1))).float(), r, p).numpy()
            else:
                ref = np.zeros(WINDOW_LEN, dtype=np.float32)
                for i in range(1, len(peaks) - 1):
                    b0, b1 = cycle_bounds(peaks, i)
                    lo, hi = max(b0, s), min(b1, s + WINDOW_LEN)
                    if lo < hi:
                        ref[lo - s:hi - s] = minmax01(ecg[b0:b1])[lo - b0:hi - b0]
            xs.append(rcg[:, s:s + WINDOW_LEN])
            refs.append(ref[None])
            ys.append(minmax01(ecg[s:s + WINDOW_LEN])[None])
            truth.append(true_ms[j])
    f = lambda v: torch.from_numpy(np.stack(v).astype(np.float32))  # noqa: E731
    return f(xs), f(refs), f(ys), np.array(truth)


def main() -> None:
    fold_i, kind = int(sys.argv[1]), sys.argv[2]
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 300
    tcn = sys.argv[4] if len(sys.argv) > 4 else "plain"
    torch.manual_seed(42)
    fold = loso_folds()[fold_i]
    train_ids, val_ids = trials_of(fold.train), trials_of(fold.val)
    trials = Trials(train_ids)
    pieces, d = [], []
    for t in train_ids:
        _, ecg, peaks = trials[t]
        pieces += [minmax01(to_len(ecg[slice(*cycle_bounds(peaks, i))])) for i in range(1, len(peaks) - 1, 5)]
        z = kde.load_cache(t)
        sl = slice(None, None, 4)
        d.append(mechanical_delay(z["phase"][(sl,) + CFG], z["ppi_ms"][(sl,) + CFG], z["starts"][sl], peaks))
    template, delta = np.mean(pieces, axis=0), float(np.median(np.concatenate(d)))
    from rri_baselines.ecg_rri import train_mean_rr
    fb = train_mean_rr(train_ids)
    t0 = time.time()
    xtr, rtr, ytr, _ = build(train_ids, kind, template, delta, fb, 160)
    xva, rva, yva, tva = build(val_ids, kind, template, delta, fb, 160)
    print(f"[{kind}] 学習 {len(xtr)} 窓 / val {len(xva)} 窓(作成 {time.time() - t0:.0f} s), δ {delta * 1000 / 200:.0f} ms", flush=True)
    model = LongTerm(tcn=tcn)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)          # 診断用に早く収束させる(論文の SGD ではない)
    g = torch.Generator().manual_seed(42)
    for k in range(steps):
        idx = torch.randint(0, len(xtr), (32,), generator=g)
        loss = torch.sqrt(F.mse_loss(model(xtr[idx], rtr[idx]), ytr[idx]))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if k % 50 == 0 or k == steps - 1:
            print(f"  歩 {k}: 学習 RMSE {loss.item():.4f} ({time.time() - t0:.0f} s)", flush=True)
    model.eval()
    res = {"fold": fold_i, "reference": kind, "steps": steps, "tcn": tcn, "train": swap_losses(model, xtr[:2000], rtr[:2000], ytr[:2000]),
           "val": swap_losses(model, xva, rva, yva)}
    with torch.no_grad():
        out = torch.cat([model(xva[i:i + 64], rva[i:i + 64]) for i in range(0, len(xva), 64)])[:, 0].numpy()
    pred = np.array([window_rr(o, fb)[0] for o in out])
    ref_pred = np.array([window_rr(r[0], fb)[0] for r in rva.numpy()])
    ok = np.isfinite(tva)
    res["val_rr_mae_ms"] = {"output": float(np.mean(np.abs(pred - tva)[ok])), "reference_only": float(np.mean(np.abs(ref_pred - tva)[ok]))}
    print(json.dumps(res, indent=1), flush=True)
    path = REPO / f"results/radarode_long/shortcut_fold{fold_i}_{kind}_{tcn}_{steps}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
