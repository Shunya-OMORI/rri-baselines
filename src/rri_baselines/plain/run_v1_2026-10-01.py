"""プレーンな網(plain/models.py)を, 提案手法(mmwave-rri)と同じ学習条件で LOSO 11 分割学習・評価する.

提案手法と同じ点: 入力は試行ごとに z スコアした RCG 50 点の 4 s 窓(200 Hz, 窓内の正規化なし), 学習窓は 0.4 s 間隔で正解のある窓だけ,
損失は窓中央 RR[秒]の L1, Adam(lr 5e-4, weight decay 1e-4), 1 エポックのウォームアップ + コサイン, バッチ 32, 20 エポック,
エポックは val の参加者の採点(protocol.score)で選ぶ, 評価は 0.2 s 間隔の全窓. 違うのは網だけ.

  .venv/bin/python -m rri_baselines.plain.run train <網> <分割...>     網 = tcn | wavenet | transformer | resnet1d
  .venv/bin/python -m rri_baselines.plain.run evaluate                  → results/plain/evaluation.json
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

from rri_baselines.plain.models import MODELS
from rri_baselines.protocol import FS, WINDOW_LEN, eval_windows, load_trial, loso_folds, rr_spanning, score, trials_of

REPO = Path(__file__).resolve().parents[3]
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS, BATCH, LR, WD, TRAIN_STEP = 20, 32, 5e-4, 1e-4, int(0.4 * FS)


def lr_at(epoch: int) -> float:                 # mmwave_rri.training.optimizer.warmup_cosine_lr と同じ
    if epoch < 1:
        return LR
    return LR * 0.5 * (1 + math.cos(math.pi * (epoch - 1) / max(1, EPOCHS - 1)))


def train_items(trial_ids: list[int], sig: dict) -> tuple[np.ndarray, np.ndarray]:
    items, ys = [], []
    for t in trial_ids:
        n, peaks = len(sig[t][0]), sig[t][1]
        for st in range(0, n - WINDOW_LEN + 1, TRAIN_STEP):
            rr = rr_spanning(peaks, st + WINDOW_LEN / 2.0, st, st + WINDOW_LEN)
            if np.isfinite(rr):
                items.append((t, st))
                ys.append(rr / FS)
    return np.array(items), np.array(ys, dtype=np.float32)


def windows(x_all: torch.Tensor, starts: np.ndarray) -> torch.Tensor:
    """つないだ信号 (L, 50) から窓 (b, 50, 800) を切り出す."""
    s0 = torch.from_numpy(starts.astype(np.int64)).to(DEV)
    return x_all[s0[:, None] + torch.arange(WINDOW_LEN, device=DEV)[None, :]].transpose(1, 2)


@torch.no_grad()
def predict_trial(model: torch.nn.Module, x: torch.Tensor, batch: int = 256) -> np.ndarray:
    model.eval()
    starts, _ = eval_windows(len(x), np.array([0.0]))
    return torch.cat([model(windows(x, starts[i:i + batch])) for i in range(0, len(starts), batch)]).cpu().numpy() * 1000.0


def train_fold(name: str, index: int) -> None:
    fold = loso_folds()[index]
    run = REPO / "runs" / f"plain_{name}" / f"fold{index:02d}_test{fold.test[0]}"
    if (run / "predictions.npz").exists():
        print(f"[skip] {run.name}", flush=True)
        return
    run.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    rng = np.random.default_rng(42)
    train_ids, val_ids, test_ids = trials_of(fold.train), trials_of(fold.val), trials_of(fold.test)
    sig = {t: (load_trial(t)["rcg_z"].astype(np.float32), load_trial(t)["peaks"]) for t in train_ids + val_ids + test_ids}
    gsig = {t: torch.from_numpy(v[0]).to(DEV) for t, v in sig.items()}
    offset, acc = {}, 0
    for t in train_ids:
        offset[t], acc = acc, acc + len(sig[t][0])
    x_all = torch.cat([gsig[t] for t in train_ids])
    items, ys = train_items(train_ids, sig)
    starts_all = np.array([offset[t] for t in items[:, 0]]) + items[:, 1]
    model = MODELS[name]().to(DEV)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    history, best = [], np.inf
    print(f"[{run.name}] {name} {n_params:,} パラメータ, 学習窓 {len(items)}", flush=True)
    for epoch in range(EPOCHS):
        t0 = time.time()
        for gp in opt.param_groups:
            gp["lr"] = lr_at(epoch)
        perm = rng.permutation(len(items))
        model.train()
        total = 0.0
        for i in range(0, len(perm), BATCH):
            idx = perm[i:i + BATCH]
            if len(idx) < 2:
                continue
            loss = (model(windows(x_all, starts_all[idx])) - torch.from_numpy(ys[idx]).to(DEV)).abs().mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss) * len(idx)
        val = score({t: predict_trial(model, gsig[t]) for t in val_ids})["mae_ms"]
        history.append({"epoch": epoch, "lr": lr_at(epoch), "train_loss": total / len(perm), "val_mae_ms": val, "sec": time.time() - t0})
        print(f"[{run.name}] epoch {epoch}: lr {lr_at(epoch):.1e} loss {total / len(perm) * 1000:.2f} ms val {val:.2f} ms ({time.time() - t0:.0f} s)", flush=True)
        if val < best:
            best = val
            torch.save(model.state_dict(), run / "best_model.pt")
        (run / "metrics.json").write_text(json.dumps({"model": name, "n_params": n_params, "history": history, "best_val_mae_ms": best}, indent=1))
    model.load_state_dict(torch.load(run / "best_model.pt", map_location=DEV))
    preds = {t: predict_trial(model, gsig[t]) for t in test_ids}
    np.savez(run / "predictions.npz", **{f"trial{t}_pred_ms": v for t, v in preds.items()})


def evaluate() -> None:
    out = {}
    for name in MODELS:
        preds = {}
        for fold in loso_folds():
            f = REPO / "runs" / f"plain_{name}" / f"fold{fold.index:02d}_test{fold.test[0]}/predictions.npz"
            if not f.exists():
                break
            z = np.load(f)
            preds.update({t: z[f"trial{t}_pred_ms"] for t in trials_of(fold.test)})
        else:
            out[name] = score(preds)
            print(f"{name}: 参加者平均 {out[name]['mae_ms']:.3f} ms")
    (REPO / "results/plain").mkdir(parents=True, exist_ok=True)
    (REPO / "results/plain/evaluation.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "train":
        for i in map(int, sys.argv[3:]):
            train_fold(sys.argv[2], i)
    else:
        evaluate()
