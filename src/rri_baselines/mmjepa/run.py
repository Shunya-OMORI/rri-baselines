"""mm-JEPA-ECG(Liu ら, AAAI-26)を MMECG 公開データ上で, 提案手法と同じ分割・目標量・採点で学習・評価する.

2 段(論文 p.38964–38967, 公式 mmJEPA.py → end2end_train.py → inference_from_train.py @ 4423ef6):
  1. pretrain: JEPA の事前学習. 文脈エンコーダ(観測 16×8 トークン)→ 予測器 → 目標エンコーダ(EMA)の 3 目標の L1
     + λ_vt × 重み付き MSE の心拍数回帰(式 (8)–(10)). 学習側 9 名の試行だけ
  2. train: エンコーダを凍結し(図 2 の雪の印, 公式の既定 = --finetune なし), HRC-DiT を ε 予測の MSE(式 (15))
     + 周波数損失(式 (16), 公式 --use-freq-loss, 重み 0.005)+ 公式の再構成 L1(重み 0.1)で学習
  3. select / test: DDIM 100 段・η=0 で ECG を生成し, 4 s 窓の中で 512 標本の生成をつないで 800 標本にし,
     ecg_rri.window_rr(NeuroKit2)で窓中央をまたぐ RR を取る

  .venv/bin/python -m rri_baselines.mmjepa.run pretrain <分割> [--epochs N] [--max-steps N] [--batch B]
  .venv/bin/python -m rri_baselines.mmjepa.run train    <分割> [--epochs N] [--max-steps N] [--batch B]
  .venv/bin/python -m rri_baselines.mmjepa.run select   <分割>     # val の窓 1 s おきで (エポック × つなぎ方) を選ぶ
  .venv/bin/python -m rri_baselines.mmjepa.run test     <分割>     # test の全窓(0.2 s おき)を予測
  .venv/bin/python -m rri_baselines.mmjepa.run evaluate            # 11 分割を採点 → results/mmjepa/summary.json
  .venv/bin/python -m rri_baselines.mmjepa.run flops               # パラメータ数と FLOPs → results/mmjepa_complexity.json
  .venv/bin/python -m rri_baselines.mmjepa.run smoke <分割> [--steps N] [--batch B] [--windows N]   # 動作確認

論文とコードの食い違い・決まらない点(選び方. 論文を優先し, 決まらない点は相手に有利な側か val)
  - 入力の点数: 論文 50 点 / end2end_train.py・inference_from_train.py の --n-vars 既定 8(自前データ用) → 50(論文, MMECG)
  - λ_vt: 論文 式 (8) と図 6(b)(0.1 > 0.05 > 0) / mmJEPA.py は重みなしの和(=1) → 0.1(図 6(b) の最良)
  - 周波数損失: 論文 式 (14)(16) で使う / 公式は --use-freq-loss が既定で切 → 入れる(重み 0.005 は公式の既定).
    公式で常に入る再構成 L1(重み 0.1)は論文に無いが残す
  - 心拍数ラベル: 論文「each patch」/ 公式は 512 標本の窓 1 つに 1 値(全トークン平均 → MLP) → 公式(32 標本 = 0.16 s の
    パッチでは 0.18–2.5 Hz の心拍は数えられない). 単位は bpm(δ = 5 BPM). ピーク検出の細部は data.py
  - 事前学習エポック: 論文図 6(a) は 50–200 で単調に改善 / mmJEPA.py docstring は -epochs 300 → 300
  - DiT のエポック: 記載なし(公式既定 1, warmup 7750 段, 50 エポックごとに保存) → 300 エポック, 50 ごとの保存を val で選ぶ
  - 推論の重み: 公式 inference は checkpoint["model"](EMA でない) → 同じ. エンコーダは student の重みを凍結(公式既定)
  - 拡散の段数: 公式 inference_from_train.py の DDIM 100 段・η=0(L167–169 のコメント). val で段数を選んだ記述は
    論文本文・コード・README・コミット履歴に無い. 付録は公開版が見つからない(arXiv 版なし, AAAI の追加資料は Underline の講演資料)
  - 学習窓のずらし幅: 記載なし → 128 標本(1 エポック 120〜187 段. 公式 log-every=155 に近い)
  - レーダの正規化: 記載なし(公式の npy 名 "UD_RMS") → 512 標本ごと・点ごとの z スコア(窓の外の統計を使わない)
  - ECG の正規化: 記載なし(DDIM の clip が [-1, 1] 前提) → 512 標本ごとに最小・最大で [-1, 1]
  - 4 s 窓(800 標本)への つなぎ方: 論文の入力は 512 標本 → single / pair / triple を val で選ぶ(下の STITCH)
"""
import argparse
import copy
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from rri_baselines.ecg_rri import train_mean_rr, window_rr
from rri_baselines.mmjepa.data import SEG, TRAIN_STEP, WindowSet, patchify
from rri_baselines.mmjepa.diffusion import T_TRAIN, Diffusion
from rri_baselines.mmjepa.masks import mm_mask
from rri_baselines.mmjepa.model import ContextEncoder, HRCDiT, MultiMaskPredictor, ecg_to_img, img_to_ecg
from rri_baselines.protocol import FS, WINDOW_LEN, eval_windows, load_trial, loso_folds, score, trials_by_subject, trials_of

REPO = Path(__file__).resolve().parents[3]
RUNS = Path(os.environ.get("MMJEPA_RUNS", REPO / "runs/mmjepa"))
DEV = torch.device(os.environ.get("MMJEPA_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
torch.backends.cuda.matmul.allow_tf32 = True          # 公式 end2end_train.py / inference_from_train.py と同じ
torch.backends.cudnn.allow_tf32 = True

# ---- 事前学習(公式 mmJEPA.py の既定. 論文と違う点は下の注)
PT_EPOCHS = 300          # 公式 docstring「python mmJEPA.py -epochs 300」. 論文図 6(a) は 50–200 で単調に改善
PT_BATCH, PT_LR, PT_WARM, PT_ETA_MIN = 128, 5e-4, 0.1, 0.2
PT_EMA, PT_WD = 0.996, (5e-4, 4e-3)
LAMBDA_VT = 0.1          # 式 (8) の λ_vt. 公式は 1(重みなしの和). 論文図 6(b) で最良の 0.1 を採る
HR_DELTA, HR_WA, HR_WB = 5.0, 1.0, 2.0      # 式 (10). 公式の既定
# ---- DiT(公式 end2end_train.py の既定)
DIT_EPOCHS = 300         # 公式は既定 1(論文に記載なし). 50 エポックごとに保存し val で選ぶ
DIT_BATCH, DIT_LR, DIT_WD, DIT_WARM, DIT_MIN_LR = 128, 5e-4, 1e-4, 7750, 1e-6
CKPT_EVERY = 50
FREQ_W, RECON_W = 0.005, 0.1
EMA_DECAY = 0.9999
# ---- 推論
DDIM_STEPS = 100         # 公式 inference_from_train.py: create_diffusion(timestep_respacing="100"), eta=0.0
GEN_OFFSETS = (0, 144, 288)          # 4 s 窓の中の 512 標本の生成位置(窓の先頭から)
STITCH = {"single": (144,), "pair": (0, 288), "triple": (0, 144, 288)}
VAL_STRIDE = 5                       # val の選択は評価窓 5 個おき(1 s おき. mmecg と同じ)


def run_dir(index: int) -> Path:
    fold = loso_folds()[index]
    d = RUNS / f"fold{index:02d}_test{fold.test[0]}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def seg_norm(rcg: torch.Tensor) -> torch.Tensor:
    """(B, 16, 50, 32) を 512 標本ごと・点ごとに z スコアへ(窓の外の統計を使わない. 学習と推論で同じ)."""
    B, L, V, P = rcg.shape
    x = rcg.permute(0, 2, 1, 3).reshape(B, V, L * P)
    x = (x - x.mean(-1, keepdim=True)) / (x.std(-1, keepdim=True) + 1e-6)
    return x.view(B, V, L, P).permute(0, 2, 1, 3).contiguous()


@torch.no_grad()
def encode(enc: ContextEncoder, rcg, pos, chunk: int = 32) -> torch.Tensor:
    """凍結エンコーダ (B, 16, 50, 32) → (B, 800, 192). 800 トークンの注意スコアが大きいので小分けにする."""
    out = torch.cat([enc(rcg[i:i + chunk], pos[i:i + chunk]) for i in range(0, len(rcg), chunk)])
    return out.reshape(out.shape[0], -1, out.shape[-1])


# ============================================================== 1. 事前学習
def weighted_hr_loss(pred, target):
    """式 (10): |ĥ−h| < δ なら W_a, そうでなければ W_b を掛けた MSE(公式 weighted_hr_loss)."""
    err = pred - target.view_as(pred)
    w = torch.where(err.abs() < HR_DELTA, torch.full_like(err, HR_WA), torch.full_like(err, HR_WB))
    return (w * err.pow(2)).mean()


def pretrain_step(student, teacher, predictor, rcg, pos, hr, rng):
    B, L, V, _ = rcg.shape
    obs, mt, mc, mb = mm_mask(B, L, V, device=rcg.device, rng=rng)
    _, z_pad = student(rcg, pos, obs)
    hats, hr_pred = predictor(z_pad, (mt, mc, mb))
    with torch.no_grad():
        full = torch.cat([teacher(rcg[i:i + 32], pos[i:i + 32]) for i in range(0, B, 32)])
    D = full.shape[-1]
    l_mask = [F.l1_loss(h, full[m].view(B, -1, D)) for h, m in zip(hats, (mt, mc, mb))]
    l_hr = weighted_hr_loss(hr_pred, hr)
    return l_mask, l_hr, float((hr_pred.view(-1) - hr).abs().mean())


def pretrain(index: int, epochs: int = PT_EPOCHS, max_steps: int | None = None, batch: int = PT_BATCH,
             out: Path | None = None) -> dict:
    fold = loso_folds()[index]
    out = out or run_dir(index)
    torch.manual_seed(42)
    rng = random.Random(42)
    tr = WindowSet(trials_of(fold.train), TRAIN_STEP, DEV)
    va = WindowSet(trials_of(fold.val), TRAIN_STEP, DEV)
    student = ContextEncoder(dropout=0.2).to(DEV)
    teacher = copy.deepcopy(student).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    predictor = MultiMaskPredictor().to(DEV)
    params = list(student.parameters()) + list(predictor.parameters())
    opt = torch.optim.AdamW(params, lr=PT_LR, weight_decay=PT_WD[0])
    steps_per_ep = len(tr) // batch                          # drop_last=True
    total = steps_per_ep * epochs if max_steps is None else max_steps
    warm = max(1, int(PT_WARM * total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: s / warm if s < warm else PT_ETA_MIN + (1 - PT_ETA_MIN)
                                              * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    ema_total = int(1.25 * total)                            # 公式: EMA と wd の時間軸はエポック数 × 1.25

    def ema_m(s):
        return PT_EMA if s < warm else PT_EMA + (s - warm) / max(1, ema_total - warm) * (1 - PT_EMA)

    def wd(s):
        return PT_WD[0] + s / max(1, ema_total - 1) * (PT_WD[1] - PT_WD[0])

    hist, step, t0 = [], 0, time.time()
    g = torch.Generator().manual_seed(42)
    for epoch in range(epochs):
        student.train(), predictor.train()
        perm = torch.randperm(len(tr), generator=g)
        acc, n = np.zeros(6), 0
        for i in range(steps_per_ep):
            if step >= total:
                break
            rcg, _, pos, hr = tr.batch(perm[i * batch:(i + 1) * batch])
            l_mask, l_hr, hr_mae = pretrain_step(student, teacher, predictor, seg_norm(rcg), pos, hr, rng)
            loss = sum(l_mask) + LAMBDA_VT * l_hr
            for gp in opt.param_groups:
                gp["weight_decay"] = wd(step)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            with torch.no_grad():
                m = ema_m(step)
                pt, ps = list(teacher.parameters()), list(student.parameters())
                torch._foreach_mul_(pt, m)
                torch._foreach_add_(pt, ps, alpha=1 - m)
            acc += [float(x) for x in l_mask] + [float(l_hr), hr_mae, float(loss)]
            n += 1
            step += 1
            if max_steps is not None and (step % 5 == 0 or step == 1):
                print(f"  step {step}: time {acc[0] / n:.4f} ch {acc[1] / n:.4f} blk {acc[2] / n:.4f} "
                      f"hr {acc[3] / n:.1f} (MAE {acc[4] / n:.1f} bpm) total {acc[5] / n:.4f}", flush=True)
        val = pretrain_val(student, teacher, predictor, va, batch, rng)
        rec = {"epoch": epoch, "step": step, "sec": time.time() - t0, "lr": sched.get_last_lr()[0],
               **dict(zip(["time", "channel", "block", "hr", "hr_mae", "total"], (acc / max(n, 1)).tolist())),
               **{f"val_{k}": v for k, v in val.items()}}
        hist.append(rec)
        print(f"[pretrain {out.name}] epoch {epoch} step {step}: loss {rec['total']:.4f} "
              f"(hr MAE {rec['hr_mae']:.1f}) val {val['total']:.4f} ({time.time() - t0:.0f} s)", flush=True)
        if step >= total:
            break
    torch.save({"student": student.state_dict(), "teacher": teacher.state_dict(),
                "predictor": predictor.state_dict(), "epochs": epochs, "steps": step}, out / "encoder.pt")
    (out / "pretrain_history.json").write_text(json.dumps(hist, indent=1))
    return hist[-1]


@torch.no_grad()
def pretrain_val(student, teacher, predictor, va, batch, rng, max_batches: int = 20) -> dict:
    student.eval(), predictor.eval()
    idx = torch.arange(len(va))[:: max(1, len(va) // (batch * max_batches))]
    acc, n = np.zeros(5), 0
    for i in range(0, len(idx), batch):
        rcg, _, pos, hr = va.batch(idx[i:i + batch])
        l_mask, l_hr, hr_mae = pretrain_step(student, teacher, predictor, seg_norm(rcg), pos, hr, rng)
        acc += [float(x) for x in l_mask] + [float(l_hr), hr_mae]
        n += 1
    acc /= max(n, 1)
    return {"total": float(acc[:3].sum() + LAMBDA_VT * acc[3]), "hr_mae": float(acc[4])}


# ============================================================== 2. HRC-DiT
def load_encoder(path: Path) -> ContextEncoder:
    enc = ContextEncoder(dropout=0.0).to(DEV)            # 公式 end2end_train.py は dp=0 で作り student を読む
    enc.load_state_dict(torch.load(path, map_location=DEV, weights_only=True)["student"])
    enc.eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc


def freq_loss(pred, target):
    """公式 frequency_domain_loss: |FFT| の L1 + 0.5 × 位相の L1(式 (16) の MAE(F(Ŷ), F(Y)))."""
    pf, tf = torch.fft.fft(pred, dim=-1), torch.fft.fft(target, dim=-1)
    return F.l1_loss(pf.abs(), tf.abs()) + 0.5 * F.l1_loss(torch.angle(pf), torch.angle(tf))


def dit_losses(dit, diff, up, ecg):
    x0 = ecg_to_img(ecg)
    t = torch.randint(0, T_TRAIN, (len(x0),), device=x0.device)
    noise = torch.randn_like(x0)
    x_t = diff.q_sample(x0, t, noise)
    eps = dit(x_t, t, up)
    l_noise = F.mse_loss(eps, noise)                      # 公式 training_losses(MSE, 576 画素すべて)
    # 公式は p_mean_variance でもう一度同じ入力を順伝播して pred_xstart を得る. DiT に dropout がないので出力は同じ
    x0_hat = img_to_ecg(diff.pred_xstart(x_t, t, eps, clip=True))
    return l_noise, freq_loss(x0_hat, ecg), F.l1_loss(x0_hat, ecg)


def train_dit(index: int, epochs: int = DIT_EPOCHS, max_steps: int | None = None, batch: int = DIT_BATCH,
              out: Path | None = None) -> dict:
    fold = loso_folds()[index]
    out = out or run_dir(index)
    torch.manual_seed(42)
    enc = load_encoder(out / "encoder.pt")
    tr = WindowSet(trials_of(fold.train), TRAIN_STEP, DEV, with_hr=False)
    dit = HRCDiT().to(DEV)
    ema = copy.deepcopy(dit).requires_grad_(False)
    diff = Diffusion()
    opt = torch.optim.AdamW(dit.parameters(), lr=DIT_LR, weight_decay=DIT_WD)
    steps_per_ep = math.ceil(len(tr) / batch)               # 公式 DataLoader は drop_last なし
    total = steps_per_ep * epochs if max_steps is None else max_steps
    warm = max(1, min(DIT_WARM, total - 1))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: s / warm if s < warm else
                                              0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm)))
                                              * (1 - DIT_MIN_LR / DIT_LR) + DIT_MIN_LR / DIT_LR)
    g = torch.Generator().manual_seed(42)
    hist, step, t0 = [], 0, time.time()
    for epoch in range(epochs):
        dit.train()
        perm = torch.randperm(len(tr), generator=g)
        acc, n = np.zeros(4), 0
        for i in range(steps_per_ep):
            if step >= total:
                break
            rcg, ecg, pos, _ = tr.batch(perm[i * batch:(i + 1) * batch])
            up = encode(enc, seg_norm(rcg), pos)
            l_noise, l_freq, l_recon = dit_losses(dit, diff, up, ecg)
            loss = l_noise + FREQ_W * l_freq + RECON_W * l_recon
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            with torch.no_grad():
                for pe, p in zip(ema.parameters(), dit.parameters()):
                    pe.mul_(EMA_DECAY).add_(p, alpha=1 - EMA_DECAY)
            acc += [float(l_noise), float(l_freq), float(l_recon), float(loss)]
            n += 1
            step += 1
            if max_steps is not None and (step % 5 == 0 or step == 1):
                print(f"  step {step}: noise {acc[0] / n:.4f} freq {acc[1] / n:.3f} recon {acc[2] / n:.4f} "
                      f"total {acc[3] / n:.4f}", flush=True)
        rec = {"epoch": epoch, "step": step, "sec": time.time() - t0, "lr": sched.get_last_lr()[0],
               **dict(zip(["noise", "freq", "recon", "total"], (acc / max(n, 1)).tolist()))}
        hist.append(rec)
        print(f"[dit {out.name}] epoch {epoch} step {step}: noise {rec['noise']:.4f} total {rec['total']:.4f} "
              f"({time.time() - t0:.0f} s)", flush=True)
        if (epoch + 1) % CKPT_EVERY == 0 or step >= total or epoch == epochs - 1:
            torch.save({"model": dit.state_dict(), "ema": ema.state_dict(), "epoch": epoch + 1, "step": step},
                       out / f"dit_ep{epoch + 1:03d}.pt")
        if step >= total:
            break
    (out / "dit_history.json").write_text(json.dumps(hist, indent=1))
    return hist[-1]


# ============================================================== 3. 推論
@torch.no_grad()
def generate(enc, dit, diff, rcg_seg, pos, seed: int, batch: int = 256) -> np.ndarray:
    """512 標本のレーダ (n, 512, 50) → 生成 ECG (n, 512). DDIM 100 段・η=0・clip. ノイズは seed で固定."""
    gen = torch.Generator(device=DEV).manual_seed(seed)
    out = []
    for i in range(0, len(rcg_seg), batch):
        x = seg_norm(patchify(rcg_seg[i:i + batch]))
        up = encode(enc, x, pos.expand(len(x), -1, -1))
        ctx = dit.embed_context(up)                        # レーダ側の k, v を 100 段で使い回す(計算結果は同じ)
        z = torch.randn(len(x), 1, 24, 24, device=DEV, generator=gen)
        img = diff.ddim_sample(lambda xx, tt: dit(xx, tt, up, ctx), z)
        out.append(img_to_ecg(img).cpu())
    return torch.cat(out).numpy()


def stitch(segs: dict[int, np.ndarray], how: str) -> np.ndarray:
    """窓の中の生成 {先頭からの位置: (n, 512)} → (n, 800).
    single: 中央 [144, 656) だけ(外は 0) / pair: [0,512) と [288,800) を重なり [288,512) で線形に切り替え /
    triple: 3 本を三角窓の重みで重ね合わせて重みの和で割る."""
    offs = STITCH[how]
    n = len(segs[offs[0]])
    acc, wsum = np.zeros((n, WINDOW_LEN)), np.zeros(WINDOW_LEN)
    if how == "pair":
        w0 = np.ones(SEG)
        w0[288:] = np.linspace(1, 0, SEG - 288)
        ws = {0: w0, 288: w0[::-1].copy()}
    elif how == "triple":
        tri = np.minimum(np.arange(1, SEG + 1), np.arange(SEG, 0, -1)).astype(float)
        ws = {o: tri for o in offs}
    else:
        ws = {144: np.ones(SEG)}
    for o in offs:
        acc[:, o:o + SEG] += segs[o] * ws[o]
        wsum[o:o + SEG] += ws[o]
    return acc / np.maximum(wsum, 1e-9)


def trial_segments(enc, dit, diff, trial_id: int, starts: np.ndarray, offsets) -> dict[int, np.ndarray]:
    tr = load_trial(trial_id)
    rcg = torch.from_numpy(tr["rcg_z"].astype(np.float32)).to(DEV)
    pos = torch.from_numpy(tr["posxyz"].astype(np.float32)).to(DEV)[None]
    ar = torch.arange(SEG, device=DEV)
    out = {}
    for k, o in enumerate(offsets):
        g = torch.from_numpy(starts + o).to(DEV)[:, None] + ar
        out[o] = generate(enc, dit, diff, rcg[g], pos, seed=42 + 1000 * trial_id + o)
    return out


def load_dit(path: Path, which: str = "model") -> HRCDiT:
    dit = HRCDiT().to(DEV)
    dit.load_state_dict(torch.load(path, map_location=DEV, weights_only=True)[which])      # 公式の推論は "model"(EMA でない)
    return dit.eval()


def select(index: int, out: Path | None = None, stride: int = VAL_STRIDE, limit: int | None = None) -> dict:
    """val の参加者の評価窓 stride 個おきで, 保存したエポック × つなぎ方(single/pair/triple)を選ぶ."""
    fold = loso_folds()[index]
    out = out or run_dir(index)
    enc, diff = load_encoder(out / "encoder.pt"), Diffusion(respace=DDIM_STEPS)
    fallback = train_mean_rr(trials_of(fold.train))
    by_subj = trials_by_subject()
    table = {}
    for ck in sorted(out.glob("dit_ep*.pt")):
        dit = load_dit(ck)
        errs = {h: {} for h in STITCH}
        for t in trials_of(fold.val):
            tr = load_trial(t)
            starts, true_ms = eval_windows(len(tr["rcg_z"]), tr["peaks"])
            sel = np.arange(0, len(starts), stride)
            sel = sel[np.isfinite(true_ms[sel])][:limit]
            segs = trial_segments(enc, dit, diff, t, starts[sel], GEN_OFFSETS)
            for h in STITCH:
                pred = np.array([window_rr(e, fallback)[0] for e in stitch(segs, h)])
                errs[h][t] = float(np.mean(np.abs(pred - true_ms[sel])))
        for h in STITCH:
            table[f"{ck.stem}/{h}"] = float(np.mean([np.mean([errs[h][t] for t in by_subj[s]]) for s in fold.val]))
        print(f"[select {out.name}] {ck.stem}: " + ", ".join(f"{h} {table[f'{ck.stem}/{h}']:.2f}" for h in STITCH),
              flush=True)
    res = {"table": table, "best": min(table, key=table.get), "stride": stride}
    (out / "select.json").write_text(json.dumps(res, indent=1))
    return res


def test(index: int, out: Path | None = None) -> None:
    fold = loso_folds()[index]
    out = out or run_dir(index)
    choice = json.loads((out / "select.json").read_text())["best"]
    ck, how = choice.split("/")
    enc, diff, dit = load_encoder(out / "encoder.pt"), Diffusion(respace=DDIM_STEPS), load_dit(out / f"{ck}.pt")
    fallback = train_mean_rr(trials_of(fold.train))
    preds, rules = {}, {}
    for t in trials_of(fold.test):
        tr = load_trial(t)
        starts, _ = eval_windows(len(tr["rcg_z"]), tr["peaks"])
        res = [window_rr(e, fallback) for e in stitch(trial_segments(enc, dit, diff, t, starts, STITCH[how]), how)]
        preds[t] = np.array([v for v, _ in res])
        rules[t] = np.array([r for _, r in res])
        print(f"[test {out.name}] trial {t}: {len(starts)} 窓", flush=True)
    np.savez(out / "predictions.npz", **{f"pred_{t}": v for t, v in preds.items()},
             **{f"rule_{t}": v for t, v in rules.items()}, choice=choice)


def evaluate() -> dict:
    preds = {}
    for d in sorted(RUNS.glob("fold*_test*")):
        f = d / "predictions.npz"
        if f.exists():
            with np.load(f) as z:
                preds.update({int(k[5:]): z[k] for k in z.files if k.startswith("pred_")})
    res = score(preds)
    out = REPO / "results/mmjepa"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(f"mm-JEPA-ECG: {res['mae_ms']:.2f} ms ({len(res['subject_mae_ms'])} 名)")
    return res


# ============================================================== パラメータ数・FLOPs
def count_flops() -> dict:
    from floppy.backends.torch_impl.hooks.universal_flop_counter import UniversalFlopCounter

    torch.manual_seed(0)
    enc, dit, pred = ContextEncoder().eval(), HRCDiT().eval(), MultiMaskPredictor().eval()
    n = lambda m: sum(p.numel() for p in m.parameters())
    params = {"encoder(推論で使う)": n(enc), "hrc_dit(推論で使う)": n(dit), "inference_total": n(enc) + n(dit),
              "predictor(事前学習だけ)": n(pred), "target_encoder_ema(事前学習だけ, エンコーダの写し)": n(enc),
              "dit_ema(学習だけ. 公式の推論は使わない)": n(dit),
              "training_total": 2 * n(enc) + n(pred) + 2 * n(dit)}

    def flops(fn) -> int:
        with torch.no_grad(), UniversalFlopCounter() as c:
            fn()
        return int(c.flops)

    x, pos = torch.randn(1, 16, 50, 32), torch.randn(1, 50, 3)
    up = torch.randn(1, 800, 192)
    xt, t = torch.randn(1, 1, 24, 24), torch.tensor([500])
    diff = Diffusion(respace=DDIM_STEPS)
    f_enc = flops(lambda: enc(x, pos))
    f_step = flops(lambda: dit(xt, t, up))                             # 公式: 毎段レーダ側の k, v を計算し直す
    f_ctx = flops(lambda: dit.embed_context(up))
    ctx = dit.embed_context(up)
    f_step_cached = flops(lambda: dit(xt, t, up, ctx))
    f_gen = flops(lambda: diff.ddim_sample(lambda a, b: dit(a, b, up), xt))   # DDIM の更新式も含めて実測
    per_gen = f_enc + f_gen
    res = {"params": params, "encoder_per_call": f_enc, "dit_step_official": f_step,
           "dit_context_kv_once": f_ctx, "dit_step_kv_cached": f_step_cached,
           "ddim100_measured_official": f_gen, "per_generation_512(encoder + DDIM 100)": per_gen}
    for how, offs in STITCH.items():
        k = len(offs)
        res[f"per_window[{how}, 生成 {k} 回]"] = k * per_gen
        res[f"per_output_second[{how}](5 窓/s)"] = 5 * k * per_gen
        res[f"per_window_kv_cached[{how}]"] = k * (f_enc + f_ctx + DDIM_STEPS * f_step_cached)
    res["continuous_ecg_per_second(512 標本の生成を重ねずに並べる)"] = per_gen * FS / SEG

    def pair_window():               # 検算: 1 窓(pair)を実際に回して数える
        for _ in range(2):
            u = enc(x, pos).reshape(1, 800, 192)
            diff.ddim_sample(lambda a, b: dit(a, b, u), xt)

    res["check_pair_window_measured"] = flops(pair_window)
    return res


# ============================================================== 動作確認
def smoke(index: int, steps: int, batch: int, windows: int) -> None:
    out = REPO / "cache/mmjepa_smoke" / f"fold{index:02d}"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"== 事前学習 {steps} 段(バッチ {batch}, {DEV})", flush=True)
    pretrain(index, epochs=1, max_steps=steps, batch=batch, out=out)
    t1 = time.time()
    print(f"== DiT {steps} 段(バッチ {batch})", flush=True)
    train_dit(index, epochs=1, max_steps=steps, batch=batch, out=out)
    t2 = time.time()
    fold = loso_folds()[index]
    t = trials_of(fold.test)[0]
    tr = load_trial(t)
    starts, true_ms = eval_windows(len(tr["rcg_z"]), tr["peaks"])
    sel = np.flatnonzero(np.isfinite(true_ms))[:windows]
    enc, diff = load_encoder(out / "encoder.pt"), Diffusion(respace=DDIM_STEPS)
    dit = load_dit(sorted(out.glob("dit_ep*.pt"))[-1])
    t3 = time.time()
    segs = trial_segments(enc, dit, diff, t, starts[sel], GEN_OFFSETS)
    t4 = time.time()
    fallback = train_mean_rr(trials_of(fold.train))
    for how in STITCH:
        ecg = stitch(segs, how)
        rr = [window_rr(e, fallback) for e in ecg]
        print(f"  {how:6s}: ECG 範囲 [{ecg.min():.2f}, {ecg.max():.2f}], RR {[round(v) for v, _ in rr]} "
              f"(規則 {[r for _, r in rr]}), 正解 {true_ms[sel].round(0).tolist()}", flush=True)
    print(f"時間: 事前学習 {t1 - t0:.1f} s, DiT {t2 - t1:.1f} s, 生成 {len(sel)} 窓 × 3 本 × DDIM {DDIM_STEPS} 段 "
          f"{t4 - t3:.1f} s ({DEV})", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pretrain", "train", "select", "test", "evaluate", "flops", "smoke"])
    ap.add_argument("folds", nargs="*", type=int)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--max-steps", type=int)
    ap.add_argument("--batch", type=int)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--windows", type=int, default=4)
    a = ap.parse_args()
    if a.cmd == "pretrain":
        for i in a.folds:
            pretrain(i, a.epochs or PT_EPOCHS, a.max_steps, a.batch or PT_BATCH)
    elif a.cmd == "train":
        for i in a.folds:
            train_dit(i, a.epochs or DIT_EPOCHS, a.max_steps, a.batch or DIT_BATCH)
    elif a.cmd == "select":
        for i in a.folds:
            select(i)
    elif a.cmd == "test":
        for i in a.folds:
            test(i)
    elif a.cmd == "evaluate":
        evaluate()
    elif a.cmd == "flops":
        res = count_flops()
        f = REPO / "results/mmjepa_complexity.json"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(res, indent=1, ensure_ascii=False))
        print(json.dumps(res, indent=1, ensure_ascii=False))
    elif a.cmd == "smoke":
        smoke(a.folds[0], a.steps, a.batch or 8, a.windows)
