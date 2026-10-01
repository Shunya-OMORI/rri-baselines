"""mm-JEPA-ECG を公式実装(無改変)で MMECG の LOSO 11 分割に通す. 本線(書き直し版 rri_baselines.mmjepa は検算用).

段(1 分割):
  export   学習 9 名 / val 1 名の試行を公式の npy に(cache/mmjepa_official/foldXX/{train,val}.npy)
  pretrain 公式 mmJEPA.py(JEPA 事前学習 + 心拍数の補助)を論文に合わせた引数で(cwd の ./data/*.npy を読む作り)
  train    公式 end2end_train.py(エンコーダ凍結, HRC-DiT)を論文に合わせた引数で. 学習中の評価(30 エポックごと,
           DDPM 1000 段で val の 1 バッチの MSE)は公式のまま走る(選択には使わない)
  select   保存した DiT の重み × 4 s 窓のつなぎ方 を, val の参加者の評価窓(1 s おき)の RR 誤差で選ぶ.
           生成は公式 inference_from_train.py(DDIM 100 段・η=0)に 4 s 窓の中の 512 標本の区間を npy で渡して行う
  test     選んだ (重み, つなぎ方) で test の参加者の全評価窓(0.2 s おき)を予測 → predictions.npz
  cleanup  大きな中間物(書き出しの npy, 選ばなかった重み)を消す

  .venv/bin/python -m rri_baselines.mmjepa_official.run all <分割> [--device cuda:0]
  .venv/bin/python -m rri_baselines.mmjepa_official.run <export|pretrain|train|select|test|cleanup> <分割>
  .venv/bin/python -m rri_baselines.mmjepa_official.run evaluate          # 11 分割を採点 → results/mmjepa_official/summary.json
  .venv/bin/python -m rri_baselines.mmjepa_official.run smoke <分割>      # 極小設定で CPU の通し確認
  .venv/bin/python -m rri_baselines.mmjepa_official.run queue             # queue/held/230_mmjepa_off_f<i>.sh を書く

4 s 窓(800 標本)の波形の作り方(公式・論文の入力は 512 標本 = 2.56 s 固定. 公式 end2end_train.py:129, :49-51 の 576 → 24×24):
  窓 [s, s+800) の中だけを使い, 窓全体を覆う位置に 512 標本の区間を置いて別々に生成し, 重なりを合成する.
  公式 inference_from_train.py も 512 標本の窓を 1 つずつ独立に生成して (N, 512) を保存するだけで, 長い記録を
  つなぐ処理は持たない. よって「区間ごとの生成」は公式の流儀そのままで, 合成だけがここで足す部分.
  窓中央の 512 標本だけを入れる形は「どれが窓中央の拍か」を人の手で絞ることになるので候補に入れない(ユーザ指示).
    pair_lin    区間 [0,512)・[288,800). 重なり [288,512) を線形に切り替え(生成 2 回/窓)
    pair_mean   同じ 2 区間. 重なりは単純平均(生成 2 回/窓)
    triple_tri  区間 0・144・288. 三角窓の重みで重ねて重みの和で割る(生成 3 回/窓)
    triple_mean 同じ 3 区間. 単純平均(生成 3 回/窓)
  生成 1 回 = エンコーダ 1 回 + DiT 100 回(DDIM). RR は ecg_rri.window_rr(NeuroKit2 0.2.13)で窓中央をまたぐ 2 ピーク.

論文と公式の食い違いの扱い(書き直し版 mmjepa/run.py の調査を引き継ぐ. 詳細は PT_ARGS / DIT_ARGS の注):
  - λ_vt = 0.1(論文 式 (8), 図 6(b) の最良): 公式は重みなしの和. 公式の -hr_weight_a / -hr_weight_b(式 (10) の W_a, W_b)を
    0.1 倍(1 → 0.1, 2 → 0.2)にすると L_vt 全体が 0.1 倍になり, λ_vt = 0.1 と厳密に同じ
  - 周波数損失(論文 式 (14)(16)): --use-freq-loss で入れる(重み 0.005 は公式の既定)
  - 事前学習 300 エポック(公式 docstring), DiT 300 エポック(記載なし. 50 ごとに保存して val で選ぶ)
  - 推論は公式どおり checkpoint["model"](EMA でない)と DDIM 100 段
"""
import argparse
import json
import os
import shutil
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from rri_baselines.ecg_rri import FALLBACK_CENTER, FALLBACK_MEAN, FALLBACK_MEDIAN, train_mean_rr, window_rr
from rri_baselines.mmjepa_official import export as ex
from rri_baselines.mmjepa_official.common import (CACHE, REPO, RUNS, SEED, assert_official_clean, run_stage,
                                                  write_manifest)
from rri_baselines.protocol import WINDOW_LEN, eval_windows, load_trial, loso_folds, score, trials_of

PY = sys.executable
LAUNCH = [PY, "-m", "rri_baselines.mmjepa_official.launch"]
SEG = 512
STITCH = {"pair_lin": (0, 288), "pair_mean": (0, 288), "triple_tri": (0, 144, 288), "triple_mean": (0, 144, 288)}
GEN_OFFSETS = (0, 144, 288)


@dataclass(frozen=True)
class Config:
    pt_epochs: int = 300          # 公式 mmJEPA.py docstring「-epochs 300」. 論文図 6(a) は 50–200 で単調に改善
    pt_batch: int = 128
    dit_epochs: int = 300         # 論文に記載なし(公式既定 1). 50 ごとの保存を val で選ぶ
    dit_batch: int = 128
    ckpt_every: int = 50          # 公式の既定
    warmup_steps: int = 7750      # 公式の既定(= 155 段 × 50 エポック)
    log_every: int = 155          # 公式の既定
    val_stride: int = 5           # val の選択は評価窓 5 個おき(1 s おき. 書き直し版・mmecg と同じ)
    infer_batch: int = 128        # 公式 inference_from_train.py の既定
    max_rows: int = 4096          # 推論の入力 npy 1 個の行数(≒ 420 MB)
    train_limit: int | None = None    # 以下は動作確認用
    valnpy_limit: int | None = None
    val_win_limit: int | None = None
    val_trials: int | None = None
    test_win_limit: int | None = None
    test_trials: int | None = None
    runs: str = str(RUNS)
    cache: str = str(CACHE)


FULL = Config()
SMOKE = Config(pt_epochs=10, pt_batch=8, dit_epochs=30, dit_batch=8, ckpt_every=15, log_every=10,
               train_limit=16, valnpy_limit=8, val_win_limit=3, val_trials=1, test_win_limit=4, test_trials=1,
               runs=str(REPO / "cache/mmjepa_official_smoke/runs"), cache=str(REPO / "cache/mmjepa_official_smoke/data"))


def pt_args(cfg: Config, device: str) -> list[str]:
    """公式 mmJEPA.py の引数(既定と同じ値も含めて全部書く). 論文 p.38966–38967 と照合済み."""
    return ["-batch_size", str(cfg.pt_batch),
            "-mask_ch", "10", "-mask_time", "2",          # 論文 p.38967「(2 time, 10 channel, 8 × 8 block)」(8×8 は公式の直書き)
            "-epochs", str(cfg.pt_epochs),
            "-lr", "5e-4", "-ema_m", "0.996",             # 公式の既定(論文に記載なし. 付録は非公開)
            "-d_model", "192", "-n_vars", "50",           # 論文 p.38966「512-sample windows with 50 channels … dmodel=192」
            "-num_workers", "1",                          # 公式では効かない(dl_kwargs に 8 が直書き). launch が上限 1 を掛ける
            "-n_layers", "2", "-pre_layers", "1", "-n_heads", "8", "-d_ff", "256",   # 論文 p.38966「two-layer … dff=256, 8 heads」
            "-ema_warmup_ratio", "0.1", "-eta_min_ratio", "0.2", "-warm", "0.1",
            "-patch_size", "32", "-stride", "32",         # 論文 p.38966「patch size and stride of 32」
            "-device", "cuda:0" if device != "cpu" else "cpu",
            "-profile_epoch", "1",
            "-hr_delta", "5.0",                           # 論文 式 (10)「δ … (e.g., 5 BPM)」
            "-hr_weight_a", "0.1", "-hr_weight_b", "0.2"]  # W_a=1, W_b=2(公式既定)× λ_vt=0.1(論文 式 (8), 図 6(b))


def dit_args(cfg: Config, train_npy: Path, val_npy: Path, enc: Path, results: Path) -> list[str]:
    """公式 end2end_train.py の引数(既定と同じ値も含めて全部書く)."""
    return ["--train-npy", str(train_npy), "--test-npy", str(val_npy),   # test-npy は学習中の評価だけに使う → val を渡す
            "--n-vars", "50",                            # 公式既定 8(自前データ用)→ 論文の 50 点
            "--patch-len", "32", "--stride", "32",
            "--encoder-ckpt", str(enc),
            "--enc-dim", "192", "--enc-layers", "2", "--enc-heads", "8", "--enc-ff", "256",
            "--epochs", str(cfg.dit_epochs), "--batch-size", str(cfg.dit_batch), "--lr", "5e-4",
            "--warmup-steps", str(cfg.warmup_steps), "--log-every", str(cfg.log_every),
            "--ckpt-every", str(cfg.ckpt_every), "--results-dir", str(results),
            # --finetune なし: 論文図 2 のエンコーダは凍結(雪の印). 公式の既定
            "--use-freq-loss", "--freq-loss-weight", "0.005",   # 論文 式 (14)(16). 公式は既定で切
            "--recon-loss-type", "l1", "--recon-loss-weight", "0.1",   # 論文に無い再構成 L1. 公式で常に入る(既定のまま)
            "--seed", str(SEED)]


def infer_args(cfg: Config, chunk: Path, enc: Path, ckpt: Path, out: Path, device: str) -> list[str]:
    return ["--test-npy", str(chunk), "--n-vars", "50", "--patch-len", "32", "--stride", "32",
            "--encoder-ckpt", str(enc),
            "--enc-dim", "192", "--enc-layers", "2", "--enc-heads", "8", "--enc-ff", "256",
            "--ckpt", str(ckpt), "--gpu-id", "0", "--batch-size", str(cfg.infer_batch),
            "--output-dir", str(out), "--seed", str(SEED), "--gen-ecg-name", "gen.npy", "--ecg-name", "ref.npy"]


# ------------------------------------------------------------------ 場所
def fold_dir(cfg: Config, index: int) -> Path:
    fold = loso_folds()[index]
    d = Path(cfg.runs) / f"fold{index:02d}_test{fold.test[0]}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def data_dir(cfg: Config, index: int) -> Path:
    return Path(cfg.cache) / f"fold{index:02d}"


def done(out: Path, stage: str) -> bool:
    f = out / "stages" / f"{stage}.json"
    return f.exists() and json.loads(f.read_text()).get("returncode") == 0


def encoder_ckpt(cfg: Config, out: Path) -> Path:
    """事前学習の最終エポックの重み. MMJEPA_COMPILE=1 のときは名前に _orig_mod. を付けた写しを返す.

    公式は mmJEPA.py で compile なしの student を保存し, end2end_train.py / inference_from_train.py では compile した
    student に load_state_dict(strict=False) で読む. compile したモジュールの名前は _orig_mod. で始まるので,
    名前が 1 つも合わず, 警告(missing/unexpected keys)だけ出して乱数のエンコーダのまま学習が進む(2026-10-01 に CPU で確認).
    既定(TORCHDYNAMO_DISABLE=1)では torch.compile が元のモジュールを返すので, この問題は起きない."""
    found = sorted((out / "pretrain/checkpoints").glob(f"*/checkpoint_epoch_{cfg.pt_epochs}.pth"))
    if not found:
        raise FileNotFoundError(f"事前学習の重み checkpoint_epoch_{cfg.pt_epochs}.pth が無い")
    src = found[-1]
    if os.environ.get("MMJEPA_COMPILE") != "1":
        return src
    import torch

    dst = out / "encoder_compiled_keys.pth"
    if not dst.exists():
        ck = torch.load(src, map_location="cpu", weights_only=False)
        ck["student_state_dict"] = {f"_orig_mod.{k}": v for k, v in ck["student_state_dict"].items()}
        torch.save(ck, dst)
    return dst


def dit_ckpts(out: Path) -> list[Path]:
    return sorted((out / "e2e").glob("*-e2e/checkpoints/epoch_*.pt"))


# ------------------------------------------------------------------ 段
def stage_export(cfg: Config, index: int) -> dict:
    fold = loso_folds()[index]
    d = data_dir(cfg, index)
    info = {}
    for name, subj, lim in (("train", fold.train, cfg.train_limit), ("val", fold.val, cfg.valnpy_limit)):
        f = d / f"{name}.npy"
        if f.exists() and f.with_suffix(".json").exists():
            info[name] = json.loads(f.with_suffix(".json").read_text())
            continue
        info[name] = ex.export_split(trials_of(subj), f, limit=lim)
        print(f"[export fold{index}] {name}: {info[name]['n']} 窓, {info[name]['bytes'] / 1e9:.2f} GB, "
              f"{info[name]['sec']:.0f} s", flush=True)
    return info


def stage_pretrain(cfg: Config, index: int, device: str) -> None:
    out = fold_dir(cfg, index)
    if done(out, "pretrain"):
        return
    d = data_dir(cfg, index)
    cwd = out / "pretrain"
    if cwd.exists():                     # 途中で止まった事前学習は最初からやり直す(公式に再開の機能は無い)
        shutil.rmtree(cwd)
    (cwd / "data").mkdir(parents=True)
    # 公式 mmJEPA.py は ./data/train_UD_RMS_all.npy と ./data/test_UD_RMS_all.npy を直書きで読む → val を test 側に置く
    (cwd / "data/train_UD_RMS_all.npy").symlink_to(d / "train.npy")
    (cwd / "data/test_UD_RMS_all.npy").symlink_to(d / "val.npy")
    run_stage("pretrain", [*LAUNCH, "mmJEPA.py", *pt_args(cfg, device)], cwd, out / "logs/pretrain.log", device, out)


def stage_train(cfg: Config, index: int, device: str) -> None:
    out = fold_dir(cfg, index)
    if done(out, "train"):
        return
    d = data_dir(cfg, index)
    res = out / "e2e"
    if res.exists():                     # 公式は results-dir の中の項目数で 000-e2e, 001-e2e … と番号を振る → 空から始める
        shutil.rmtree(res)
    res.mkdir(parents=True)
    args = dit_args(cfg, d / "train.npy", d / "val.npy", encoder_ckpt(cfg, out), res)
    run_stage("train", [*LAUNCH, "end2end_train.py", *args], out, out / "logs/train.log", device, out)
    log = (next(res.glob("*-e2e")) / "log.txt").read_text(encoding="utf-8", errors="replace")
    if "missing keys" in log or "unexpected keys" in log or "Loaded encoder weights." not in log:
        raise RuntimeError("end2end_train.py がエンコーダの重みを読めていない(log.txt に missing/unexpected keys)")


def infer_rows(cfg: Config, trials: list[int], stride: int, offsets, win_limit: int | None):
    """[(試行, 窓番号, 区間の開始位置)] と, 試行ごとの (評価窓の番号, 正解 RR[ms]). 窓の外の信号は使わない."""
    rows, truth = [], {}
    for t in trials:
        tr = load_trial(t)
        starts, true_ms = eval_windows(len(tr["rcg_z"]), tr["peaks"])
        sel = np.arange(0, len(starts), stride)
        if stride > 1:                   # val の選択: 正解のある窓だけ(test は採点の都合で全窓)
            sel = sel[np.isfinite(true_ms[sel])]
        sel = sel[:win_limit]
        truth[t] = (sel, true_ms[sel])
        rows += [(t, int(w), int(starts[w] + o)) for w in sel for o in offsets]
        assert all(starts[w] + max(offsets) + SEG <= starts[w] + WINDOW_LEN for w in sel)
    return rows, truth


def generate(cfg: Config, out: Path, tag: str, rows, enc: Path, ckpt: Path, device: str) -> dict:
    """rows の区間を公式 inference_from_train.py で生成 → {(試行, 窓, 区間の開始位置): (512,)}. 結果は gen.npz に残す."""
    gdir = out / "infer" / tag
    cached = gdir / "gen.npz"
    if cached.exists():
        z = np.load(cached)
        return {tuple(r): g for r, g in zip(z["rows"].tolist(), z["gen"])}
    chunks = ex.export_infer(rows, Path(cfg.cache) / out.name / "infer" / tag, cfg.max_rows)
    all_rows, all_gen, sec = [], [], 0.0
    for c in chunks:
        co = gdir / Path(c["path"]).stem
        sec += run_stage(f"infer_{tag.replace('/', '_')}_{co.name}",
                         [*LAUNCH, "inference_from_train.py", *infer_args(cfg, Path(c["path"]), enc, ckpt, co, device)],
                         out, out / "logs/infer.log", device, out)
        g = np.load(co / "gen.npy")
        assert len(g) == len(c["rows"]), (len(g), len(c["rows"]))
        all_rows.append(c["rows"])
        all_gen.append(g.astype(np.float32))
        Path(c["path"]).unlink()                 # 入力の npy はすぐ消す(1 個 ≒ 420 MB)
        (co / "ref.npy").unlink(missing_ok=True)
        (co / "gen.npy").unlink()                # gen.npz にまとめる
    rows_a, gen_a = np.concatenate(all_rows), np.concatenate(all_gen)
    np.savez(cached, rows=rows_a, gen=gen_a, sec=sec, n_generations=len(rows_a))
    return {tuple(r): g for r, g in zip(rows_a.tolist(), gen_a)}


def stitch(gens: dict, t: int, w: int, start: int, how: str) -> np.ndarray:
    """4 s 窓 [start, start+800) の中の区間の生成を合成して (800,)."""
    offs = STITCH[how]
    acc, wsum = np.zeros(WINDOW_LEN), np.zeros(WINDOW_LEN)
    if how == "pair_lin":
        w0 = np.ones(SEG)
        w0[288:] = np.linspace(1, 0, SEG - 288)
        ws = {0: w0, 288: w0[::-1].copy()}
    elif how == "triple_tri":
        tri = np.minimum(np.arange(1, SEG + 1), np.arange(SEG, 0, -1)).astype(float)
        ws = {o: tri for o in offs}
    else:                                # *_mean
        ws = {o: np.ones(SEG) for o in offs}
    for o in offs:
        acc[o:o + SEG] += gens[(t, w, start + o)] * ws[o]
        wsum[o:o + SEG] += ws[o]
    return acc / np.maximum(wsum, 1e-9)


def window_preds(gens: dict, truth: dict, how: str, fallback: float) -> dict:
    preds = {}
    for t, (sel, _) in truth.items():
        tr = load_trial(t)
        starts, _ = eval_windows(len(tr["rcg_z"]), tr["peaks"])
        preds[t] = [window_rr(stitch(gens, t, int(w), int(starts[w]), how), fallback) for w in sel]
    return preds


def stage_select(cfg: Config, index: int, device: str) -> dict:
    out = fold_dir(cfg, index)
    f = out / "select.json"
    if f.exists():
        return json.loads(f.read_text())
    fold = loso_folds()[index]
    val_trials = trials_of(fold.val)[:cfg.val_trials]
    rows, truth = infer_rows(cfg, val_trials, cfg.val_stride, GEN_OFFSETS, cfg.val_win_limit)
    fallback = train_mean_rr(trials_of(fold.train))
    enc = encoder_ckpt(cfg, out)
    table, detail = {}, {}
    for ck in dit_ckpts(out):
        gens = generate(cfg, out, f"val/{ck.stem}", rows, enc, ck, device)
        for how in STITCH:
            p = window_preds(gens, truth, how, fallback)
            err = {t: float(np.mean(np.abs(np.array([v for v, _ in p[t]]) - truth[t][1]))) for t in truth}
            key = f"{ck.stem}/{how}"
            table[key] = float(np.mean(list(err.values())))          # val は 1 名 → 試行平均の平均
            rules = np.array([r for t in p for _, r in p[t]])
            detail[key] = {"trial_mae": err, "rule_share": {n: float(np.mean(rules == c)) for n, c in
                           (("center", FALLBACK_CENTER), ("median", FALLBACK_MEDIAN), ("train_mean", FALLBACK_MEAN))}}
        print(f"[select fold{index}] {ck.stem}: " + ", ".join(f"{h} {table[f'{ck.stem}/{h}']:.2f}" for h in STITCH),
              flush=True)
    best = min(table, key=table.get)
    res = {"table": table, "best": best, "detail": detail, "val_subject": fold.val, "val_trials": val_trials,
           "val_stride": cfg.val_stride, "n_val_windows": int(sum(len(s) for s, _ in truth.values())),
           "generations_per_window": {h: len(o) for h, o in STITCH.items()}, "ddim_steps": 100,
           "fallback_ms": fallback}
    f.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    return res


def stage_test(cfg: Config, index: int, device: str) -> None:
    out = fold_dir(cfg, index)
    if (out / "predictions.npz").exists():
        return
    fold = loso_folds()[index]
    ck_name, how = json.loads((out / "select.json").read_text())["best"].split("/")
    ck = next(c for c in dit_ckpts(out) if c.stem == ck_name)
    test_trials = trials_of(fold.test)[:cfg.test_trials]
    rows, truth = infer_rows(cfg, test_trials, 1, STITCH[how], cfg.test_win_limit)
    gens = generate(cfg, out, f"test/{ck_name}_{how}", rows, encoder_ckpt(cfg, out), ck, device)
    p = window_preds(gens, truth, how, train_mean_rr(trials_of(fold.train)))
    np.savez(out / "predictions.npz", **{f"pred_{t}": np.array([v for v, _ in p[t]]) for t in p},
             **{f"rule_{t}": np.array([r for _, r in p[t]]) for t in p},
             **{f"true_{t}": truth[t][1] for t in p}, choice=f"{ck_name}/{how}",
             generations_per_window=len(STITCH[how]))
    for t in p:
        v = np.array([x for x, _ in p[t]])
        ok = np.isfinite(truth[t][1])
        print(f"[test fold{index}] trial {t}: {len(v)} 窓, MAE {np.mean(np.abs(v[ok] - truth[t][1][ok])):.2f} ms",
              flush=True)


def stage_cleanup(cfg: Config, index: int) -> None:
    """書き出しの npy と, 選ばなかった DiT の重み・途中の事前学習の重みを消す(選んだ重みと最終エンコーダは残す)."""
    out = fold_dir(cfg, index)
    if not (out / "predictions.npz").exists():
        raise RuntimeError("test の予測がまだ無いので消さない")
    shutil.rmtree(data_dir(cfg, index), ignore_errors=True)
    shutil.rmtree(Path(cfg.cache) / out.name, ignore_errors=True)
    keep = json.loads((out / "select.json").read_text())["best"].split("/")[0]
    for c in dit_ckpts(out):
        if c.stem != keep:
            c.unlink()
    for c in (out / "pretrain/checkpoints").glob("*/checkpoint_epoch_*.pth"):
        if c.name != f"checkpoint_epoch_{cfg.pt_epochs}.pth":
            c.unlink()
    for name in ("train_UD_RMS_all.npy", "test_UD_RMS_all.npy"):
        (out / "pretrain/data" / name).unlink(missing_ok=True)


def run_all(cfg: Config, index: int, device: str) -> None:
    assert_official_clean()
    out = fold_dir(cfg, index)
    fold = loso_folds()[index]
    write_manifest(out, {"fold": fold._asdict(), "config": asdict(cfg), "device": device,
                         "official_args": {"mmJEPA.py": pt_args(cfg, device),
                                           "end2end_train.py": dit_args(cfg, Path("<train.npy>"), Path("<val.npy>"),
                                                                        Path("<encoder>"), Path("<results>")),
                                           "inference_from_train.py": infer_args(cfg, Path("<chunk>"), Path("<encoder>"),
                                                                                 Path("<ckpt>"), Path("<out>"), device)},
                         "stitch": {k: list(v) for k, v in STITCH.items()}})
    t0 = time.time()
    info = stage_export(cfg, index)
    (out / "export.json").write_text(json.dumps(info, indent=1))
    t1 = time.time()
    stage_pretrain(cfg, index, device)
    t2 = time.time()
    stage_train(cfg, index, device)
    t3 = time.time()
    stage_select(cfg, index, device)
    t4 = time.time()
    stage_test(cfg, index, device)
    t5 = time.time()
    times = {"export": t1 - t0, "pretrain": t2 - t1, "train": t3 - t2, "select": t4 - t3, "test": t5 - t4}
    (out / "times.json").write_text(json.dumps(times, indent=1))
    print(f"[fold{index}] 時間 " + ", ".join(f"{k} {v / 60:.1f} 分" for k, v in times.items()), flush=True)


def evaluate(cfg: Config) -> dict:
    preds = {}
    for d in sorted(Path(cfg.runs).glob("fold*_test*")):
        f = d / "predictions.npz"
        if f.exists():
            with np.load(f) as z:
                preds.update({int(k[5:]): z[k] for k in z.files if k.startswith("pred_")})
    res = score(preds)
    out = REPO / "results/mmjepa_official"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(f"mm-JEPA-ECG(公式実装): {res['mae_ms']:.2f} ms ({len(res['subject_mae_ms'])} 名)")
    return res


QUEUE_HELD = Path("/home/shunya/research/queue/held")


def write_queue(cfg: Config) -> None:
    """1 分割を学習から評価まで通すシェルスクリプトを 11 本(queue/held/230_mmjepa_off_f<i>.sh). 投入は主担当."""
    for i, fold in enumerate(loso_folds()):
        args_pt = " ".join(pt_args(cfg, "cuda:0"))
        args_dit = " ".join(dit_args(cfg, Path("<cache>/train.npy"), Path("<cache>/val.npy"),
                                     Path("<pretrain>/checkpoint_epoch_300.pth"), Path("<run>/e2e")))
        args_inf = " ".join(infer_args(cfg, Path("<chunk>.npy"), Path("<encoder>"), Path("<epoch_XXX.pt>"),
                                       Path("<out>"), "cuda:0"))
        body = f"""#!/bin/bash
# mm-JEPA-ECG(Liu ら, AAAI-26)公式実装 lanyangyang/mmJEPA-ECG @ 4423ef6 を無改変で, MMECG LOSO の分割 {i}
# (test 参加者 {fold.test[0]}, val 参加者 {fold.val[0]}, 学習 9 名). 書き出し → 事前学習 → DiT → val で (重み × つなぎ方) を選択 → test → 後片づけ.
# 途中で止めたら同じスクリプトを流し直せば, 終わった段は飛ばす(公式の学習自体は途中再開できないので, その段は最初から).
#
# 公式に渡す引数(rri_baselines/mmjepa_official/run.py の pt_args / dit_args / infer_args が正本. 根拠は同ファイルの注):
#   mmJEPA.py {args_pt}
#   end2end_train.py {args_dit}
#   inference_from_train.py {args_inf}
#
# 資源: GPU 1 枚(MMJEPA_DEVICE=cuda:0, CUDA_VISIBLE_DEVICES で選ぶ). RAM は学習 npy(≦2.5 GB)の読み込みで一時 5.4 GB(s7 実測).
# GPU メモリ: 公式のバッチ 128 では PatchTST の注意スコア (128, 8, 800, 800) を materialize するので, 活性化だけで
# 事前学習 ≒ 12 GB・凍結エンコーダの順伝播 ≒ 9 GB(CPU の最大 RSS から 93.5 / 71 MB/標本, 2026-10-01 s7 実測).
# RTX 3060 Ti(8 GB)では走らない. A6000 / A5000 で回す(16 GB 未満の GPU なら始めずに止める)
set -e
cd /home/shunya/research/rri-baselines
GPU_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "${{CUDA_VISIBLE_DEVICES:-0}}" | head -1)
[ "${{GPU_MB:-0}}" -ge 16000 ] || {{ echo "GPU メモリ ${{GPU_MB}} MB: 公式のバッチ 128 は入らないので止める"; exit 1; }}
# 手元の WSL(16 GB)を落とさない: 空きが 7 GB 未満なら始めない
awk '/MemAvailable/ {{exit !($2 > 7*1024*1024)}}' /proc/meminfo || {{ echo "空きメモリが 7 GB 未満なので止める"; exit 1; }}
export PYTHONDONTWRITEBYTECODE=1 MMJEPA_DEVICE=cuda:0 MMJEPA_MAX_WORKERS=1
.venv/bin/python -m rri_baselines.mmjepa_official.run all {i} --device cuda:0
.venv/bin/python -m rri_baselines.mmjepa_official.run cleanup {i}
"""
        p = QUEUE_HELD / f"230_mmjepa_off_f{i}.sh"
        p.write_text(body)
        p.chmod(0o755)
        print(p)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["all", "export", "pretrain", "train", "select", "test", "cleanup", "evaluate",
                                    "smoke", "queue"])
    ap.add_argument("folds", nargs="*", type=int)
    ap.add_argument("--device", default=os.environ.get("MMJEPA_DEVICE", "cuda:0"))
    a = ap.parse_args()
    if a.cmd == "smoke":
        for i in a.folds:
            run_all(SMOKE, i, "cpu")
    elif a.cmd == "evaluate":
        evaluate(FULL)
    elif a.cmd == "queue":
        write_queue(FULL)
    else:
        for i in a.folds:
            {"all": lambda: run_all(FULL, i, a.device), "export": lambda: stage_export(FULL, i),
             "pretrain": lambda: stage_pretrain(FULL, i, a.device), "train": lambda: stage_train(FULL, i, a.device),
             "select": lambda: stage_select(FULL, i, a.device), "test": lambda: stage_test(FULL, i, a.device),
             "cleanup": lambda: stage_cleanup(FULL, i)}[a.cmd]()
