"""MIBINET を MMECG 上で, 提案手法と同じ分割・目標量・採点で学習・評価する(specs/mibinet2023.md §7.3 の本線).

- 入力: 4 s 窓の RCG 50 点を 500 Hz に補間した (50, 2000). 窓ごと・点ごとに z スコア. 論文は 1 チャネルだが 50 点を入れる(情報が増える側)
- 出力: 窓の中心をまたぐ拍の RR[秒] 1 個(4 s 窓には論文の 7 拍が入らない). 後処理なしを本線, 中央値 3(提案手法と同じ)を併記
- 損失・最適化・学習率: 論文どおり(式 (2)(3) の重み, Adam, バッチ 1024, 学習率 0.007 から Fig. 9 の区分的な指数減衰).
  エポック数 EPOCHS に合わせて減衰の時間軸を縮める(論文は 200)
- 拡張(式 (1)): 中心の RR が 0.9〜1.1 s の学習窓に, x(t) + x(t + α), α ~ U(450, 550) ms を 1 本ずつ足す. 正解は R 波の和集合から作り直す
- 学習窓は 0.4 s 間隔(提案手法と同じ). エポックは val の参加者の窓中央 RR の MAE で選ぶ

  .venv/bin/python -m rri_baselines.mibinet.run train <分割...> | evaluate
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.signal import resample_poly

from rri_baselines.mibinet.model import MIBINet, mibinet_loss
from rri_baselines.protocol import FS, WINDOW_LEN, eval_windows, load_trial, loso_folds, rr_spanning, score, trials_of

REPO = Path(__file__).resolve().parents[3]
RUNS = REPO / "runs/mibinet"
OUT = REPO / "results/mibinet"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
UP = 2.5                                     # 200 → 500 Hz
WIN = int(WINDOW_LEN * UP)                   # 2,000 点
EPOCHS = int(os.environ.get("MIBINET_EPOCHS", 200))
BATCH, TRAIN_STEP = 1024, int(0.4 * FS)
AUG_LO, AUG_HI = int(0.45 * FS), int(0.55 * FS)


def lr_at(epoch: int) -> float:
    """Fig. 9: 0〜80 エポックは 40 エポックで半分, 80 で 7e-5 に落とし, 以後 40 エポックで 1/10. EPOCHS に合わせて縮める."""
    e = epoch * 200 / EPOCHS
    return 0.007 * 0.5 ** (e / 40) if e < 80 else 7e-5 * 0.1 ** ((e - 80) / 40)


def load_up(trial_id: int) -> tuple[np.ndarray, np.ndarray]:
    t = load_trial(trial_id)
    return resample_poly(t["rcg_z"], 5, 2, axis=0).astype(np.float32), t["peaks"]


def normalize(x: torch.Tensor) -> torch.Tensor:
    """(b, 2000, 50) → (b, 50, 2000). 窓ごと・点ごとの z スコア."""
    x = (x - x.mean(1, keepdim=True)) / (x.std(1, keepdim=True) + 1e-6)
    return x.transpose(1, 2)


def train_items(trial_ids: list[int], sig: dict, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """(試行の番号, 開始[200 Hz 標本], 拡張のずれ[200 Hz 標本, 0 = なし]) と正解 RR[秒]."""
    items, ys = [], []
    for t in trial_ids:
        n, peaks = len(sig[t][0]) / UP, sig[t][1]
        for st in range(0, int(n) - WINDOW_LEN + 1, TRAIN_STEP):
            rr = rr_spanning(peaks, st + WINDOW_LEN / 2.0, st, st + WINDOW_LEN)
            if not np.isfinite(rr):
                continue
            items.append((t, st, 0))
            ys.append(rr / FS)
            if 0.9 <= rr / FS <= 1.1:
                a = int(rng.integers(AUG_LO, AUG_HI + 1))
                if st + a + WINDOW_LEN <= n:
                    union = np.unique(np.concatenate([peaks, peaks - a]))
                    rr2 = rr_spanning(union, st + WINDOW_LEN / 2.0, st, st + WINDOW_LEN)
                    if np.isfinite(rr2):
                        items.append((t, st, a))
                        ys.append(rr2 / FS)
    return np.array(items), np.array(ys, dtype=np.float32)


def batch_input(items: np.ndarray, x_all: torch.Tensor, offset: dict) -> torch.Tensor:
    """試行をつないだ 500 Hz の信号 x_all (L, 50) から窓 (b, 50, 2000) を一括で切り出す. 拡張の窓はずらした窓を足す."""
    base = torch.tensor([offset[t] for t in items[:, 0]], device=DEV)
    ar = torch.arange(WIN, device=DEV)[None, :]
    s0 = base + torch.from_numpy(np.round(items[:, 1] * UP).astype(np.int64)).to(DEV)
    x = x_all[s0[:, None] + ar]
    a = torch.from_numpy(items[:, 2]).to(DEV)
    s1 = base + torch.from_numpy(np.round((items[:, 1] + items[:, 2]) * UP).astype(np.int64)).to(DEV)
    x = x + (a > 0).float()[:, None, None] * x_all[s1[:, None] + ar]
    return normalize(x)


@torch.no_grad()
def predict_trial(model: MIBINet, x: torch.Tensor, n: int, batch: int = 512) -> np.ndarray:
    """評価の窓(0.2 s 間隔)ごとの RR[ms]."""
    model.eval()
    starts, _ = eval_windows(n, np.array([0.0]))
    out = []
    for i in range(0, len(starts), batch):
        s0 = torch.from_numpy(np.round(starts[i:i + batch] * UP).astype(np.int64)).to(DEV)
        xb = x[s0[:, None] + torch.arange(WIN, device=DEV)[None, :]]
        out.append(model(normalize(xb)).squeeze(1))
    return torch.cat(out).cpu().numpy() * 1000.0


def median3(v: np.ndarray) -> np.ndarray:
    p = np.pad(v, 1, mode="edge")
    return np.median(np.stack([p[:-2], p[1:-1], p[2:]]), axis=0)


def train_fold(index: int) -> None:
    fold = loso_folds()[index]
    run = RUNS / f"fold{index:02d}_test{fold.test[0]}"
    if (run / "predictions.npz").exists():
        print(f"[skip] {run.name}", flush=True)
        return
    run.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    rng = np.random.default_rng(42)
    train_ids, val_ids, test_ids = trials_of(fold.train), trials_of(fold.val), trials_of(fold.test)
    sig = {t: load_up(t) for t in train_ids + val_ids + test_ids}
    gsig = {t: torch.from_numpy(v[0]).to(DEV) for t, v in sig.items()}
    offset, acc = {}, 0
    for t in train_ids:
        offset[t], acc = acc, acc + len(sig[t][0])
    x_all = torch.cat([gsig[t] for t in train_ids])
    lens = {t: int(round(len(v[0]) / UP)) for t, v in sig.items()}
    model = MIBINet().to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr_at(0), betas=(0.9, 0.999), eps=1e-7)
    history, best = [], np.inf
    for epoch in range(EPOCHS):
        t0 = time.time()
        for gp in opt.param_groups:
            gp["lr"] = lr_at(epoch)
        items, ys = train_items(train_ids, sig, rng)            # 拡張のずれはエポックごとに引き直す
        perm = rng.permutation(len(items))
        model.train()
        total = 0.0
        for i in range(0, len(perm), BATCH):
            idx = perm[i:i + BATCH]
            if len(idx) < 2:
                continue
            pred = model(batch_input(items[idx], x_all, offset)).squeeze(1)
            loss = mibinet_loss(pred, torch.from_numpy(ys[idx]).to(DEV))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss) * len(idx)
        val = score({t: predict_trial(model, gsig[t], lens[t]) for t in val_ids})["mae_ms"]
        history.append({"epoch": epoch, "lr": lr_at(epoch), "train_loss": total / len(perm), "val_mae_ms": val,
                        "n_train": len(items), "sec": time.time() - t0})
        print(f"[{run.name}] epoch {epoch}: lr {lr_at(epoch):.1e} loss {total / len(perm):.5f} val {val:.2f} ms "
              f"({time.time() - t0:.0f} s)", flush=True)
        if val < best:
            best = val
            torch.save(model.state_dict(), run / "best_model.pt")
        (run / "metrics.json").write_text(json.dumps({"history": history, "best_val_mae_ms": best}, indent=1))
    model.load_state_dict(torch.load(run / "best_model.pt", map_location=DEV))
    preds = {t: predict_trial(model, gsig[t], lens[t]) for t in test_ids}
    np.savez(run / "predictions.npz", **{f"trial{t}_pred_ms": v for t, v in preds.items()})


def evaluate() -> None:
    preds = {}
    for fold in loso_folds():
        z = np.load(RUNS / f"fold{fold.index:02d}_test{fold.test[0]}/predictions.npz")
        preds.update({t: z[f"trial{t}_pred_ms"] for t in trials_of(fold.test)})
    result = {**score(preds), "median3": score({t: median3(v) for t, v in preds.items()})}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "evaluation.json").write_text(json.dumps(result, indent=1))
    print(f"MIBINET(4 s 窓): 参加者平均 {result['mae_ms']:.3f} ms(中央値 3: {result['median3']['mae_ms']:.3f} ms)")


if __name__ == "__main__":
    if sys.argv[1] == "train":
        for i in map(int, sys.argv[2:]):
            train_fold(i)
    else:
        evaluate()
