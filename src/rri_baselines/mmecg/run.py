"""MMECG を MMECG 公開データ上で, 提案手法と同じ分割・目標量・採点で学習・評価する.

- 学習(p.279 §4.2.3): 640 標本の窓を 30 標本ずつずらして切り出す. Adam 1e-3, バッチ 64, μ 則 256 値の交差エントロピー,
  teacher forcing. 損失を取る位置は既定で窓の末尾 128 標本(「step equals to 128 (input time series size − receptive
  field size)」). --span all で全 640 標本(先頭は因果的なゼロ埋め)にもできる(下の「決まらない点」)
- 入力: 窓の中の RCG 50 点(rcg_z)を, エンコーダに入れる 640 標本ごと・点ごとに z スコアにし直す(窓の外の統計を使わない).
  位置は posxyz (50, 3)
- 正解 ECG: 試行ごとに中央値を引き, |x| の 99.9 パーセンタイルで割って [-1, 1] に切る(式 (14) の「normalized in -1 < x < 1」)
- 推論: 評価の 4 s 窓 [s, s+800) ごとに独立に 800 標本を 1 標本ずつ生成する. 条件 h は窓の中の 2 つの 640 標本
  A = [s, s+640) と B = [s+160, s+800) のエンコーダ出力をつなぐ. RR は ecg_rri.window_rr(NeuroKit2)で取る
- 論文で決まらない推論の選択(val の参加者の窓 1 s おきで選ぶ):
  生成前の過去の ECG(pad: 学習時と同じゼロ埋め / median: 学習側の中央値を 512 標本), h のつなぎ方
  (switch640 / switch400 / avg), 出力の取り方(argmax / sample)
- エポックは val の参加者の窓 1 s おきの RR 誤差(推論は pad・switch640・argmax)で選ぶ
- --span noar: 過去の ECG を使わない形(2026-10-01 追加). 原文どおりの teacher forcing では TCN が正解の過去 512 標本
  だけで次の標本を当てる近道を覚え, h を無視する(h を他の窓と入れ替えても損失が変わらない. 生成の val RR 誤差 436 ms).
  noar は過去の入力をゼロにして全 640 標本で損失を取り, 推論は 1 回の順伝播で 800 標本を出す(1 エポック相当で val 12.5 ms)
- MMECG_PATIENCE(既定 8): val が続けて改善しなければ打ち切るエポック数

  .venv/bin/python -m rri_baselines.mmecg.run train [--span 128|all] <分割...>
  .venv/bin/python -m rri_baselines.mmecg.run evaluate [--span 128|all]
  .venv/bin/python -m rri_baselines.mmecg.run smoke <分割> <ステップ数> [バッチ]    # 動作確認
  .venv/bin/python -m rri_baselines.mmecg.run flops                                 # パラメータ数と FLOPs
"""
import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from rri_baselines.ecg_rri import train_mean_rr, window_rr
from rri_baselines.mmecg.model import (COND, MMECG, RECEPTIVE, SEG, class_value, generate, mu_law_classes,
                                       mu_law_decode)
from rri_baselines.protocol import WINDOW_LEN, eval_windows, load_trial, loso_folds, score, trials_of

REPO = Path(__file__).resolve().parents[3]
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = int(os.environ.get("MMECG_EPOCHS", 30))
PATIENCE = int(os.environ.get("MMECG_PATIENCE", 8))
BATCH, LR, TRAIN_STEP = 64, 1e-3, 30
ENC_BATCH = int(os.environ.get("MMECG_ENC_BATCH", 64))      # 推論でエンコーダに一度に入れる窓の数
GEN_BATCH = int(os.environ.get("MMECG_GEN_BATCH", 2048))    # 自己回帰で一度に進める窓の数
OFFSET_B = WINDOW_LEN - SEG                                 # 160: 後ろ側のエンコーダ窓の開始
VAL_EVERY = 5                                               # val の窓を 1 s おき(5 窓に 1 つ)に間引く
CONTEXTS, STITCHES, DECODES = ("pad", "median"), ("switch640", "switch400", "avg"), ("argmax", "sample")
DEFAULT = {"context": "pad", "stitch": "switch640", "decode": "argmax"}
NOAR_DEFAULT = {"context": "none", "stitch": "switch640", "decode": "argmax"}   # context none = 過去を使わない 1 回の順伝播


def run_root(span: str) -> Path:
    return REPO / "runs" / f"mmecg_s{span}"


# ---- データ --------------------------------------------------------------------------------------------
def normalize_ecg(ecg: np.ndarray) -> np.ndarray:
    e = ecg - np.median(ecg)
    return np.clip(e / np.percentile(np.abs(e), 99.9), -1.0, 1.0)


def load(trial_id: int) -> dict:
    t = load_trial(trial_id)
    return {"rcg": torch.from_numpy(t["rcg_z"].astype(np.float32)),
            "cls": mu_law_classes(torch.from_numpy(normalize_ecg(t["ecg"]).astype(np.float32))),
            "pos": torch.from_numpy(t["posxyz"].astype(np.float32)), "peaks": t["peaks"], "n": len(t["rcg_z"])}


def zscore(x: torch.Tensor) -> torch.Tensor:
    """(b, 50, L) を窓ごと・点ごとに z スコア."""
    return (x - x.mean(-1, keepdim=True)) / (x.std(-1, keepdim=True) + 1e-6)


class Pool:
    """試行をつないで GPU に置き, (試行の番号, 開始) の組から 640 標本の窓を一括で切り出す."""

    def __init__(self, data: dict, trial_ids: list[int]) -> None:
        self.index = {t: i for i, t in enumerate(trial_ids)}
        self.offset = torch.tensor(np.cumsum([0] + [data[t]["n"] for t in trial_ids[:-1]]), device=DEV)
        self.rcg = torch.cat([data[t]["rcg"] for t in trial_ids]).to(DEV)
        self.cls = torch.cat([data[t]["cls"] for t in trial_ids]).to(DEV)
        self.pos = torch.stack([data[t]["pos"] for t in trial_ids]).to(DEV)
        self.ar = torch.arange(SEG, device=DEV)

    def batch(self, items: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ti = torch.from_numpy(np.array([self.index[t] for t in items[:, 0]])).to(DEV)
        s0 = self.offset[ti] + torch.from_numpy(items[:, 1]).to(DEV)
        idx = s0[:, None] + self.ar
        return zscore(self.rcg[idx].transpose(1, 2)), self.pos[ti], self.cls[idx]


def train_items(data: dict, trial_ids: list[int]) -> np.ndarray:
    return np.array([(t, s) for t in trial_ids for s in range(0, data[t]["n"] - SEG + 1, TRAIN_STEP)], dtype=np.int64)


def tf_loss(model: MMECG, x, pos, cls, span: str) -> torch.Tensor:
    """teacher forcing の交差エントロピー. 位置 j の入力は 1 つ前の標本の値(先頭は 0 = ゼロ埋め)."""
    x_prev = torch.cat([torch.zeros_like(cls[:, :1], dtype=torch.float32), class_value(cls[:, :-1])], 1)
    if span == "noar":
        x_prev = torch.zeros_like(x_prev)
    logits = model(x, pos, x_prev)
    k = SEG - RECEPTIVE if span == "128" else SEG
    return F.cross_entropy(logits[:, :, -k:], cls[:, -k:])


# ---- 推論 ----------------------------------------------------------------------------------------------
@torch.no_grad()
def window_conditions(model: MMECG, trial: dict, starts: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """各評価窓の中の 2 つのエンコーダ窓 A = [s, s+640), B = [s+160, s+800) の条件 (n, 4, 640) ずつ(CPU に置く)."""
    model.eval()
    rcg, pos = trial["rcg"].to(DEV), trial["pos"].to(DEV)
    ar = torch.arange(SEG, device=DEV)
    hs = {0: [], OFFSET_B: []}
    for i in range(0, len(starts), ENC_BATCH):
        s = torch.from_numpy(starts[i:i + ENC_BATCH]).to(DEV)
        for off in hs:
            x = zscore(rcg[(s + off)[:, None] + ar].transpose(1, 2))
            hs[off].append(model.encoder(x, pos.expand(len(s), -1, -1)).cpu())
    return torch.cat(hs[0]), torch.cat(hs[OFFSET_B])


def stitch(ha: torch.Tensor, hb: torch.Tensor, mode: str) -> torch.Tensor:
    """(n, 4, 640) ×2 → 評価窓 800 標本の条件 (n, 4, 800). B の位置 j は窓の時刻 160 + j."""
    k = {"switch640": SEG, "switch400": WINDOW_LEN // 2}.get(mode)
    if k is not None:
        return torch.cat([ha[:, :, :k], hb[:, :, k - OFFSET_B:]], 2)
    mid = 0.5 * (ha[:, :, OFFSET_B:] + hb[:, :, :SEG - OFFSET_B])
    return torch.cat([ha[:, :, :OFFSET_B], mid, hb[:, :, SEG - OFFSET_B:]], 2)


@torch.no_grad()
def generate_windows(model: MMECG, h: torch.Tensor, context: str, decode: str, median_value: float) -> np.ndarray:
    """条件 (n, 4, 800) → 生成した ECG (n, 800)[-1, 1]."""
    model.eval()
    g = torch.Generator(device=DEV).manual_seed(42)
    out = []
    for i in range(0, len(h), GEN_BATCH):
        hb = h[i:i + GEN_BATCH].to(DEV)
        if context == "none":                  # noar: 過去をゼロにした 1 回の順伝播
            logits = model.decoder(torch.zeros(hb.shape[0], hb.shape[2], device=DEV), hb)
            c = (torch.distributions.Categorical(logits=logits.transpose(1, 2)).sample() if decode == "sample"
                 else logits.argmax(1))
            out.append(mu_law_decode(c).cpu())
            continue
        ctx = None if context == "pad" else torch.full((len(hb), RECEPTIVE), median_value, device=DEV)
        out.append(mu_law_decode(generate(model.decoder, hb, ctx, decode == "sample", g)).cpu())
    return torch.cat(out).numpy()


def rr_of(ecg_windows: np.ndarray, fallback_ms: float) -> tuple[np.ndarray, np.ndarray]:
    out = [window_rr(w, fallback_ms) for w in ecg_windows]
    return np.array([v for v, _ in out]), np.array([r for _, r in out])


def predict(model, data, trial_ids, setting, fallback_ms, median_value, every: int = 1, cond=None):
    """試行ごとの RR[ms](eval_windows の並び. every > 1 なら間引いた窓だけ)と規則の内訳."""
    preds, rules = {}, []
    for t in trial_ids:
        starts, _ = eval_windows(data[t]["n"], data[t]["peaks"])
        starts = starts[::every]
        ha, hb = cond[t] if cond is not None else window_conditions(model, data[t], starts)
        ecg = generate_windows(model, stitch(ha, hb, setting["stitch"]), setting["context"], setting["decode"],
                               median_value)
        preds[t], r = rr_of(ecg, fallback_ms)
        rules.append(r)
    return preds, np.concatenate(rules)


def score_subsampled(preds: dict, data: dict, every: int) -> float:
    """間引いた窓での参加者平均 MAE(protocol.score と同じ集計: 試行平均 → 参加者平均)."""
    by_subject: dict[int, list[float]] = {}
    for t, p in preds.items():
        _, true_ms = eval_windows(data[t]["n"], data[t]["peaks"])
        true_ms = true_ms[::every]
        ok = np.isfinite(true_ms)
        by_subject.setdefault(int(load_trial(t)["subject_id"]), []).append(float(np.mean(np.abs(p[ok] - true_ms[ok]))))
    return float(np.mean([np.mean(v) for v in by_subject.values()]))


# ---- 学習 ----------------------------------------------------------------------------------------------
def train_fold(index: int, span: str) -> None:
    fold = loso_folds()[index]
    run = run_root(span) / f"fold{index:02d}_test{fold.test[0]}"
    if (run / "predictions.npz").exists():
        print(f"[skip] {run.name}", flush=True)
        return
    run.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    rng = np.random.default_rng(42)
    train_ids, val_ids, test_ids = trials_of(fold.train), trials_of(fold.val), trials_of(fold.test)
    data = {t: load(t) for t in train_ids + val_ids + test_ids}
    pool = Pool(data, train_ids)
    items = train_items(data, train_ids)
    fallback_ms = train_mean_rr(train_ids)
    median_value = float(class_value(pool.cls.median()))
    model = MMECG().to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    history, best = [], np.inf
    for epoch in range(EPOCHS):
        t0 = time.time()
        model.train()
        perm, total = rng.permutation(len(items)), 0.0
        for i in range(0, len(perm), BATCH):
            loss = tf_loss(model, *pool.batch(items[perm[i:i + BATCH]]), span)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss) * len(perm[i:i + BATCH])
        t1 = time.time()
        preds, rules = predict(model, data, val_ids, NOAR_DEFAULT if span == "noar" else DEFAULT, fallback_ms,
                               median_value, VAL_EVERY)
        val = score_subsampled(preds, data, VAL_EVERY)
        history.append({"epoch": epoch, "train_ce": total / len(perm), "val_mae_ms": val,
                        "val_center_share": float(np.mean(rules == 0)), "train_sec": t1 - t0, "val_sec": time.time() - t1})
        print(f"[{run.name}] epoch {epoch}: ce {total / len(perm):.4f} val {val:.2f} ms "
              f"(中央をまたぐ組 {np.mean(rules == 0):.0%}) 学習 {t1 - t0:.0f} s 検証 {time.time() - t1:.0f} s", flush=True)
        if val < best:
            best = val
            torch.save(model.state_dict(), run / "best_model.pt")
        (run / "metrics.json").write_text(json.dumps({"history": history, "best_val_mae_ms": best}, indent=1))
        if epoch - min(range(len(history)), key=lambda e: history[e]["val_mae_ms"]) >= PATIENCE:
            print(f"[{run.name}] val が {PATIENCE} エポック改善しないので打ち切り", flush=True)
            break
    model.load_state_dict(torch.load(run / "best_model.pt", map_location=DEV))
    select_and_predict(model, data, val_ids, test_ids, fallback_ms, median_value, run, span)


def select_and_predict(model, data, val_ids, test_ids, fallback_ms, median_value, run: Path, span: str = "128") -> None:
    """推論の選択(過去の ECG・h のつなぎ方・出力の取り方)を val の窓 1 s おきで選び, test の全窓を予測する."""
    cond = {}
    for t in val_ids:
        starts, _ = eval_windows(data[t]["n"], data[t]["peaks"])
        cond[t] = window_conditions(model, data[t], starts[::VAL_EVERY])
    table = []
    for c, s, d in itertools.product(("none",) if span == "noar" else CONTEXTS, STITCHES, DECODES):
        setting = {"context": c, "stitch": s, "decode": d}
        preds, rules = predict(model, data, val_ids, setting, fallback_ms, median_value, VAL_EVERY, cond)
        table.append({**setting, "val_mae_ms": score_subsampled(preds, data, VAL_EVERY),
                      "center_share": float(np.mean(rules == 0))})
        print(f"[{run.name}] 選択 {setting}: val {table[-1]['val_mae_ms']:.2f} ms", flush=True)
    chosen = min(table, key=lambda r: r["val_mae_ms"])
    setting = {k: chosen[k] for k in DEFAULT}
    t0 = time.time()
    preds, rules = predict(model, data, test_ids, setting, fallback_ms, median_value)
    (run / "selection.json").write_text(json.dumps(
        {"table": table, "chosen": chosen, "fallback_ms": fallback_ms, "median_value": median_value,
         "test_rule_share": {k: float(np.mean(rules == v)) for k, v in (("center", 0), ("median", 1), ("train_mean", 2))},
         "test_sec": time.time() - t0}, indent=1, ensure_ascii=False))
    np.savez(run / "predictions.npz", **{f"trial{t}_pred_ms": v for t, v in preds.items()})


def evaluate(span: str) -> None:
    preds = {}
    for fold in loso_folds():
        z = np.load(run_root(span) / f"fold{fold.index:02d}_test{fold.test[0]}/predictions.npz")
        preds.update({t: z[f"trial{t}_pred_ms"] for t in trials_of(fold.test)})
    result = score(preds)
    out = REPO / "results" / f"mmecg_s{span}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "evaluation.json").write_text(json.dumps(result, indent=1))
    print(f"MMECG(損失 {span}): 参加者平均 {result['mae_ms']:.3f} ms")


# ---- 動作確認 ------------------------------------------------------------------------------------------
def smoke(index: int, steps: int, batch: int, span: str) -> None:
    """数十ステップだけ学習して損失の下がり方・1 ステップの時間・メモリ, 1 窓と一括の生成時間を出す."""
    fold = loso_folds()[index]
    torch.manual_seed(42)
    rng = np.random.default_rng(42)
    train_ids = trials_of(fold.train)[:4]
    data = {t: load(t) for t in train_ids}
    pool = Pool(data, train_ids)
    items = train_items(data, train_ids)
    model = MMECG().to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    fixed = items[rng.choice(len(items), batch, replace=False)]
    torch.cuda.reset_peak_memory_stats() if DEV.type == "cuda" else None
    for step in range(steps):
        model.train()
        loss = tf_loss(model, *pool.batch(items[rng.choice(len(items), batch, replace=False)]), span)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % max(1, steps // 10) == 0 or step == steps - 1:
            model.eval()
            with torch.no_grad():
                fl = float(tf_loss(model, *pool.batch(fixed), span))
            print(f"step {step}: 学習バッチ ce {float(loss):.4f}  固定バッチ ce {fl:.4f}", flush=True)
    if DEV.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(5):
        loss = tf_loss(model, *pool.batch(items[rng.choice(len(items), batch, replace=False)]), span)
        opt.zero_grad()
        loss.backward()
        opt.step()
    if DEV.type == "cuda":
        torch.cuda.synchronize()
        print(f"学習 1 ステップ(バッチ {batch}): {(time.time() - t0) / 5 * 1000:.0f} ms, "
              f"最大メモリ {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB", flush=True)
    t = train_ids[0]
    starts, _ = eval_windows(data[t]["n"], data[t]["peaks"])
    for n in (1, 256):
        if DEV.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        ha, hb = window_conditions(model, data[t], starts[:n])
        t1 = time.time()
        ecg = generate_windows(model, stitch(ha, hb, "switch640"), "pad", "argmax", 0.0)
        t2 = time.time()
        rr, _ = rr_of(ecg, 800.0)
        print(f"窓 {n} 個: エンコーダ {t1 - t0:.2f} s, 自己回帰 800 標本 {t2 - t1:.2f} s, NeuroKit {time.time() - t2:.2f} s. "
              f"生成 ECG の範囲 [{ecg.min():.2f}, {ecg.max():.2f}], RR 例 {rr[:3].round(0)}", flush=True)


def count_flops() -> dict:
    from floppy.backends.torch_impl.hooks.universal_flop_counter import UniversalFlopCounter

    torch.manual_seed(0)
    model = MMECG().eval()
    n_params = {"total": sum(p.numel() for p in model.parameters()),
                "encoder": sum(p.numel() for p in model.encoder.parameters()),
                "decoder": sum(p.numel() for p in model.decoder.parameters())}

    def flops(fn) -> int:
        with torch.no_grad(), UniversalFlopCounter() as c:
            fn()
        return int(c.flops)

    x, pos = torch.randn(1, 50, SEG), torch.randn(1, 50, 3)
    enc = flops(lambda: model.encoder(x, pos))
    h1 = torch.randn(1, COND, 1)
    step_cached = (flops(lambda: generate(model.decoder, torch.randn(1, COND, 101)))
                   - flops(lambda: generate(model.decoder, torch.randn(1, COND, 1)))) / 100
    # キャッシュなし: 各時刻で受容野 512 標本ぶんの TCN を計算し直し, 最後の出力だけ使う
    step_naive = flops(lambda: model.decoder(torch.zeros(1, RECEPTIVE), torch.randn(1, COND, RECEPTIVE)))
    tf_640 = flops(lambda: model.decoder(torch.zeros(1, SEG), torch.randn(1, COND, SEG)))
    window_cached = 2 * enc + WINDOW_LEN * step_cached
    window_naive = 2 * enc + WINDOW_LEN * step_naive
    res = {"params": n_params,
           "encoder_per_call": enc, "tcn_step_cached": step_cached, "tcn_step_naive_rf512": step_naive,
           "tcn_teacher_forcing_640": tf_640,
           "per_window_cached": window_cached, "per_window_naive": window_naive,
           "per_output_second_cached(5 窓/s)": 5 * window_cached, "per_output_second_naive(5 窓/s)": 5 * window_naive,
           "continuous_ecg_per_second_cached(200 標本 + エンコーダ 200/128 回)": 200 * step_cached + 200 / 128 * enc,
           "training_forward_per_sample(エンコーダ 1 回 + TCN 640 並列)": enc + tf_640}
    # 検算: 800 標本の生成を実際に回して数える(エンコーダ 2 回 + 生成)
    res["check_window_cached_measured"] = flops(lambda: (model.encoder(x, pos), model.encoder(x, pos),
                                                         generate(model.decoder, torch.randn(1, COND, WINDOW_LEN))))
    return res


if __name__ == "__main__":
    args = sys.argv[1:]
    span = "128"
    if "--span" in args:
        i = args.index("--span")
        span = args[i + 1]
        del args[i:i + 2]
    assert span in ("128", "all", "noar"), span
    if args[0] == "train":
        for i in map(int, args[1:]):
            train_fold(i, span)
    elif args[0] == "evaluate":
        evaluate(span)
    elif args[0] == "smoke":
        smoke(int(args[1]), int(args[2]), int(args[3]) if len(args) > 3 else 8, span)
    elif args[0] == "flops":
        res = count_flops()
        out = REPO / "results/mmecg_complexity.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=1, ensure_ascii=False))
        print(json.dumps(res, indent=1, ensure_ascii=False))
