"""Wu, Sakamoto ら (2019) を MMECG 上で, 提案手法と同じ分割・目標量・採点で学習・評価する.

載せ方(literature/docs/baselines/specs/pseudo_pulse_wu2019.md §8):
- 50 点 → 1 本: 試行ごとの相関行列の第 1 固有ベクトルで重み付け和(論文のビーム形成に倣う. ラベルは使わない). 符号は重みの和が正になる向き
- 200 Hz → 416.67 Hz に再標本化して, 6.48 s = 2,700 点の φ(t) を 54×50 に並べる(論文の形のまま)
- φ(t) は試行の実信号を使う(4 s 窓の外も見る. 相手に有利な側で, 提案手法より多い情報). 試行の端の外は 0
- φ(t) ごとに [0, 1] へ min-max 正規化(論文 §III-C. 範囲が書かれていないので窓ごと)
- ラベル: W1 は R 波で 1・R–R の中点で 0 の三角波, W2 は S 波で 0・中点で 1(論文は 0〜100). S 波は R 波の後 80 ms の ECG の最小
- RR: 窓の中の W1 の極大と W2 の極小のうち, 窓の中央をはさむ 2 つの間隔を R・S で平均する.
  片方しか無ければその間隔, 両方無ければ窓の中の極値の間隔の中央値, それも無ければ学習側の RR の平均(除外しない).
  極値の最小間隔・平滑化・S 波を使うかは論文に無いので, 分割ごとに val の誤差で選ぶ(相手に有利な側)
- 学習: 損失は MSE(MATLAB の回帰層), Adam 1e-3, バッチ 256, 20 エポック, 出力の時刻は 20 ms ごと(1 エポック約 60 万点).
  エポックは val の参加者の三角波の二乗誤差で選ぶ(学習の目的と同じ量. 極値の条件は学習後に val で選ぶ)

  .venv/bin/python -m rri_baselines.wu2019.run train <分割...> | evaluate
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.signal import find_peaks, resample_poly

from rri_baselines.protocol import FS, WINDOW_LEN, eval_windows, load_trial, loso_folds, score, trials_of
from rri_baselines.wu2019.model import PHI_LEN, WuCNN

REPO = Path(__file__).resolve().parents[3]
RUNS = REPO / "runs/wu2019"
OUT = REPO / "results/wu2019"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HALF = PHI_LEN // 2
EPOCHS, LR, BATCH, TRAIN_STEP = 20, 1e-3, 256, 4
S_SEARCH = 16                 # R 波の後 80 ms で S 波を探す


def beamform(rcg: np.ndarray) -> np.ndarray:
    _, vecs = np.linalg.eigh(rcg.T @ rcg / len(rcg))
    v = vecs[:, -1] * np.sign(vecs[:, -1].sum())
    return rcg @ v


def triangle(marks: np.ndarray, n: int, peak_value: float) -> np.ndarray:
    """印の時刻で peak_value, 隣の印との中点で 1 − peak_value の三角波 (n,). 最初と最後の印の外は NaN."""
    w = np.full(n, np.nan)
    for a, b in zip(marks[:-1], marks[1:]):
        t = np.arange(a, b)
        dist = np.minimum(t - a, b - t) / ((b - a) / 2.0)          # 印で 0, 中点で 1
        w[a:b] = peak_value + (1 - 2 * peak_value) * dist
    return w


def prepare(trial_id: int) -> dict:
    trial = load_trial(trial_id)
    x = resample_poly(beamform(trial["rcg_z"].astype(np.float64)), 25, 12).astype(np.float32)
    peaks = trial["peaks"].astype(int)
    s_waves = np.array([p + int(np.argmin(trial["ecg"][p:p + S_SEARCH])) for p in peaks if p + S_SEARCH <= len(trial["ecg"])])
    n = len(trial["rcg_z"])
    return {"x": np.pad(x, HALF), "n": n, "w1": triangle(peaks, n, 1.0), "w2": triangle(s_waves, n, 0.0),
            "peaks": trial["peaks"]}


def gather_phi(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """出力の時刻 t[200 Hz 標本](b,) の φ (b, 2700). x は両端に HALF 点の 0 を足した 416.67 Hz の信号."""
    pos = torch.round(t.float() * 25 / 12).long() + HALF
    phi = x[pos[:, None] + torch.arange(-HALF, HALF, device=x.device)[None, :]]
    lo, hi = phi.min(1, keepdim=True).values, phi.max(1, keepdim=True).values
    return (phi - lo) / (hi - lo + 1e-8)


@torch.no_grad()
def predict_series(model: WuCNN, prep: dict, batch: int = 4096) -> np.ndarray:
    x = torch.from_numpy(prep["x"]).to(DEV)
    t = torch.arange(prep["n"], device=DEV)
    return torch.cat([model(gather_phi(x, t[i:i + batch])) for i in range(0, len(t), batch)]).cpu().numpy()


def window_rr_ms(w: np.ndarray, n: int, fallback_ms: float, use_s: bool = True,
                 min_dist_sec: float = 0.3, smooth: int = 1) -> np.ndarray:
    """試行の (W1, W2) 系列 (n, 2) から, 評価の窓ごとの RR[ms].
    極値の最小間隔と平滑化(移動平均の幅[標本])は論文に無いので, 分割ごとに val で選ぶ(DECODE_GRID). use_s=False は R 波(W1)だけ."""
    if smooth > 1:
        k = np.ones(smooth) / smooth
        w = np.stack([np.convolve(w[:, 0], k, "same"), np.convolve(w[:, 1], k, "same")], 1)
    dist = int(min_dist_sec * FS)
    r = find_peaks(w[:, 0], distance=dist)[0]
    s = find_peaks(-w[:, 1], distance=dist)[0] if use_s else np.array([], dtype=int)
    starts, _ = eval_windows(n, np.array([0.0]))
    out = np.empty(len(starts))
    for i, st in enumerate(starts):
        c, end = st + WINDOW_LEN / 2.0, st + WINDOW_LEN
        spans, inner = [], []
        for marks in (r, s):
            m = marks[(marks >= st) & (marks < end)]
            k = np.searchsorted(m, c)
            if 0 < k < len(m):
                spans.append(m[k] - m[k - 1])
            inner.extend(np.diff(m))
        out[i] = np.mean(spans) / FS * 1000.0 if spans else (np.median(inner) / FS * 1000.0 if inner else fallback_ms)
    return out


# (S 波も使うか, 最小間隔[秒], 平滑化の幅[標本]). 論文どおりは (True, 0.3, 1) に最も近い
DECODE_GRID = [(u, d, sm) for u in (True, False) for d in (0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8)
               for sm in (1, 11, 21, 41)]


def fold_predictions(model: WuCNN, trial_ids: list[int], preps: dict, fallback_ms: float, use_s: bool = True) -> dict:
    model.eval()
    return {t: window_rr_ms(predict_series(model, preps[t]), preps[t]["n"], fallback_ms, use_s) for t in trial_ids}


def val_mse(model: WuCNN, trial_ids: list[int], preps: dict) -> float:
    """val の参加者の三角波の二乗誤差(学習の目的と同じ量). エポックの選択に使う(復号の条件に依らない)."""
    model.eval()
    err, n = 0.0, 0
    for t in trial_ids:
        w = predict_series(model, preps[t])
        y = np.stack([preps[t]["w1"], preps[t]["w2"]], 1)
        ok = np.isfinite(y).all(1)
        err += float(np.sum((w[ok] - y[ok]) ** 2))
        n += int(ok.sum()) * 2
    return err / n


def train_fold(index: int) -> None:
    fold = loso_folds()[index]
    run = RUNS / f"fold{index:02d}_test{fold.test[0]}"
    if (run / "predictions.npz").exists():
        print(f"[skip] {run.name}", flush=True)
        return
    run.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    np.random.seed(42)
    train_ids, val_ids, test_ids = trials_of(fold.train), trials_of(fold.val), trials_of(fold.test)
    preps = {t: prepare(t) for t in train_ids + val_ids + test_ids}
    fallback_ms = float(np.mean(np.concatenate([np.diff(preps[t]["peaks"]) for t in train_ids]))) / FS * 1000.0
    # 学習の点: 試行をつないだ信号の上の (信号の添字, 出力の時刻, ラベル)
    xs, offs, ts, ys = [], [0], [], []
    for k, t in enumerate(train_ids):
        p = preps[t]
        xs.append(p["x"])
        tt = np.arange(0, p["n"], TRAIN_STEP)
        y = np.stack([p["w1"][tt], p["w2"][tt]], 1)
        ok = np.isfinite(y).all(1)
        ts.append(np.stack([np.full(ok.sum(), k), tt[ok]], 1))
        ys.append(y[ok])
        offs.append(offs[-1] + len(p["x"]))
    x_all = torch.from_numpy(np.concatenate(xs)).to(DEV)
    off = torch.tensor(offs[:-1], device=DEV)
    ts, ys = torch.from_numpy(np.concatenate(ts)).to(DEV), torch.from_numpy(np.concatenate(ys)).float().to(DEV)
    model = WuCNN().to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    g = torch.Generator(device="cpu").manual_seed(42)
    history, best = [], np.inf
    for epoch in range(EPOCHS):
        t0, model_loss = time.time(), 0.0
        model.train()
        perm = torch.randperm(len(ts), generator=g).to(DEV)
        for i in range(0, len(perm), BATCH):
            idx = perm[i:i + BATCH]
            k, tt = ts[idx, 0], ts[idx, 1]
            # 試行ごとの信号の先頭位置をずらして, つないだ信号の上で φ を切り出す
            pos = torch.round(tt.float() * 25 / 12).long() + HALF + off[k]
            phi = x_all[pos[:, None] + torch.arange(-HALF, HALF, device=DEV)[None, :]]
            lo, hi = phi.min(1, keepdim=True).values, phi.max(1, keepdim=True).values
            loss = torch.mean((model((phi - lo) / (hi - lo + 1e-8)) - ys[idx]) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            model_loss += float(loss) * len(idx)
        val = val_mse(model, val_ids, preps)
        history.append({"epoch": epoch, "train_mse": model_loss / len(perm), "val_mse": val, "sec": time.time() - t0})
        print(f"[{run.name}] epoch {epoch}: train {model_loss / len(perm):.4f} val {val:.4f} ({time.time() - t0:.0f} s)", flush=True)
        if val < best:
            best = val
            torch.save(model.state_dict(), run / "best_model.pt")
    model.load_state_dict(torch.load(run / "best_model.pt", map_location=DEV))
    model.eval()
    series = {t: predict_series(model, preps[t]) for t in val_ids + test_ids}
    np.savez_compressed(run / "series.npz", **{f"trial{t}": v.astype(np.float32) for t, v in series.items()})
    val_mae = {c: score({t: window_rr_ms(series[t], preps[t]["n"], fallback_ms, *c) for t in val_ids})["mae_ms"]
               for c in DECODE_GRID}
    chosen = min(val_mae, key=val_mae.get)
    preds = {t: window_rr_ms(series[t], preps[t]["n"], fallback_ms, *chosen) for t in test_ids}
    paper = {t: window_rr_ms(series[t], preps[t]["n"], fallback_ms, True, 0.3, 1) for t in test_ids}
    np.savez(run / "predictions.npz", **{f"trial{t}_pred_ms": v for t, v in preds.items()},
             **{f"trial{t}_pred_paper_ms": v for t, v in paper.items()})
    (run / "metrics.json").write_text(json.dumps({"history": history, "best_val_mse": best, "fallback_ms": fallback_ms,
                                                  "decode_chosen": chosen,
                                                  "decode_val_mae_ms": {str(c): v for c, v in val_mae.items()}}, indent=1))


def redecode() -> None:
    """保存した val・test の出力系列から, 極値の条件を DECODE_GRID で選び直す(学習し直さない)."""
    for fold in loso_folds():
        run = RUNS / f"fold{fold.index:02d}_test{fold.test[0]}"
        z, m = np.load(run / "series.npz"), json.loads((run / "metrics.json").read_text())
        n = {t: len(load_trial(t)["rcg_z"]) for t in trials_of(fold.val) + trials_of(fold.test)}
        fb = m["fallback_ms"]
        val_mae = {c: score({t: window_rr_ms(z[f"trial{t}"], n[t], fb, *c) for t in trials_of(fold.val)})["mae_ms"]
                   for c in DECODE_GRID}
        chosen = min(val_mae, key=val_mae.get)
        old = np.load(run / "predictions.npz")
        np.savez(run / "predictions.npz",
                 **{f"trial{t}_pred_ms": window_rr_ms(z[f"trial{t}"], n[t], fb, *chosen) for t in trials_of(fold.test)},
                 **{k: old[k] for k in old.files if k.endswith("_paper_ms")})
        m["decode_chosen"], m["decode_val_mae_ms"] = chosen, {str(c): v for c, v in val_mae.items()}
        (run / "metrics.json").write_text(json.dumps(m, indent=1))
        print(run.name, chosen, round(val_mae[chosen], 1), flush=True)


def evaluate() -> None:
    preds, preds_r = {}, {}
    for fold in loso_folds():
        z = np.load(RUNS / f"fold{fold.index:02d}_test{fold.test[0]}/predictions.npz")
        preds.update({t: z[f"trial{t}_pred_ms"] for t in trials_of(fold.test)})
        preds_r.update({t: z[f"trial{t}_pred_paper_ms"] for t in trials_of(fold.test)})
    result = {**score(preds), "paper_decode": score(preds_r)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "evaluation.json").write_text(json.dumps(result, indent=1))
    print(f"Wu・Sakamoto 2019: 参加者平均 {result['mae_ms']:.3f} ms(極値の条件を論文どおり(S 波も使う・0.3 s・平滑化なし)に固定すると {result['paper_decode']['mae_ms']:.3f} ms)")
    print(" ".join(f"s{s}:{v:.2f}" for s, v in result["subject_mae_ms"].items()))


if __name__ == "__main__":
    if sys.argv[1] == "train":
        for i in map(int, sys.argv[2:]):
            train_fold(i)
    elif sys.argv[1] == "redecode":
        redecode()
    else:
        evaluate()
