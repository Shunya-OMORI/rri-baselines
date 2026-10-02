"""radarODE-MTL の PPI 以外の 2 タスク(ECG 波形の再構成・R 波位置 = アンカー)を, 論文 Table I と同じ指標で評価する.

  python scripts/radarode_task_eval.py <run 名> [--windows all|labeled] [--stride 40] [--thr 0.5] [--batch 64] [--ckpt auto|last]
  --ckpt auto: 波形は best_ECG_shape.pt, アンカーは best_Anchor.pt(無ければ best_model.pt = PPI の val 最良)
  --ckpt last: 両タスクとも resume.pt の最終エポックの重み(参考. 出力は <run 名>_tasks_last.md)

  run 名: radarode_ew_wsst_official200 → 公開分割(test = 試行 75〜85. 集計は試行平均)
          radarode_ew_wsst200          → LOSO 11 分割(fold ディレクトリを回る. 集計は 試行 → 参加者 → 11 名平均)
  環境変数は launch_wsst.sh と同じものが要る(PYTHONPATH=src, RRI_DATA_ROOT, RADARODE_ROOT, RADARODE_SST, RADARODE_TARGET).
  run の config.json の sst・target と環境変数が食い違えば止まる.

論文の指標(原文で確認済み, 指示書より):
  波形  : "RMSE, PCC, and R² for the recovered single-cycle ECG pieces". 正解の ECG 片は 200 点に再標本化.
  アンカー: "an absolute timing error and a missed detected rate (MDR)". 150 ms は MDR の記述にだけ現れる.

このスクリプトの読み方:
  波形  : 1 窓 = 対象の拍(窓の中央をまたぐ R–R)の ECG 片 200 点. 学習の損失(shapeLoss)は正解の片を
          片ごとに min-max で [0, 1] にしてから MSE を取るので, 模型の出力は [0, 1] の尺度にある.
          RMSE は (a) その [0, 1] の尺度(公開実装の shapeMetric の MSE と同じ尺度) と
          (b) 正解の片の min・max で出力を ECG の単位に戻した値(= (a) × 片の振幅幅) の両方を出す.
          PCC と R² は正解・出力に同じ一次変換をかけても変わらないので 1 通り.
          R² = 1 − Σ(y − ŷ)² / Σ(y − ȳ)²(片ごと).
          R3 の対照として「入力を見ずに学習側の平均の片([0, 1] 尺度)を出すだけ」の値を併記する.
  アンカー: 出力 (800,) を窓ごとに min-max で [0, 1] にし(公開実装の anchorMetric と同じ正規化),
          scipy.signal.find_peaks(高さ ≥ thr, 間隔 ≥ 0.3 s = 60 標本)で R 波の候補を取る.
          正解の R 波(Anchor ラベルの 1 の位置)ごとに最も近い候補を対応づけ, |差| ≤ 150 ms なら一致として
          timing error に入れ, 一致が無ければ missed. MDR = missed / 正解の R 波の数.
          150 ms は論文の MDR の記述に合わせたもの. timing error の許容窓は論文に無い.
予測は cache/radarode_task_preds/ に別名で保存する(runs/ には書かない).
"""
import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from scipy.signal import find_peaks

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import rri_baselines.radarode.train as rtrain  # noqa: E402
from rri_baselines.protocol import FS, WINDOW_LEN, load_trial, loso_folds, trials_by_subject, trials_of  # noqa: E402
from rri_baselines.radarode.data import SST_VARIANT, TARGET, CenterBeatWindows, down_sample, target_index, target_rr  # noqa: E402

MATCH_MS = 150.0
MATCH = int(round(MATCH_MS / 1000 * FS))          # 30 標本
MIN_DIST = int(round(0.3 * FS))                   # 60 標本
MS_PER_SAMPLE = 1000.0 / FS
THR_SWEEP = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
PAPER = {"rmse": 0.125, "pcc": 79.7, "r2": 0.63, "timing_ms": 9.7, "mdr": 5.51}


# ---------------------------------------------------------------- 正解(SST を読まずに data.py と同じ式で作る)
def gt_piece(ecg: np.ndarray, peaks: np.ndarray, s: int) -> np.ndarray | None:
    k = target_index(peaks, s)
    if not 0 < k < len(peaks):
        return None
    return down_sample(ecg[peaks[k - 1]:peaks[k]]).astype(np.float64)


def gt_rpeaks(peaks: np.ndarray, s: int) -> np.ndarray:
    return peaks[(peaks >= s) & (peaks < s + WINDOW_LEN)] - s


def minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = x.min(axis=-1, keepdims=True), x.max(axis=-1, keepdims=True)
    return (x - lo) / np.where(hi - lo < 1e-12, 1.0, hi - lo)


# ---------------------------------------------------------------- 推論
def load_model(ckpt: Path):
    rtrain.set_device = lambda *_: None          # LibMTL の set_device は CUDA_VISIBLE_DEVICES を '0' で上書きするので止める
    with tempfile.TemporaryDirectory() as d:     # 学習器の run_dir は一時ディレクトリ(runs/ に何も作らない)
        g = torch.Generator()
        g.manual_seed(42)
        trainer = rtrain.build_trainer(Path(d), 200, g)
    st = torch.load(ckpt, map_location=trainer.device, weights_only=False)
    trainer.model.load_state_dict(st["model"] if ckpt.name == "resume.pt" else st)      # resume.pt は最終エポックの再開点
    trainer.model.eval()
    return trainer


@torch.no_grad()
def infer(ckpt: Path, trials: list[int], stride: int, batch: int, cache: Path) -> dict:
    """試行ごとに {items: (n, 2) [試行, 開始], shape: (n, 200), anchor: (n, 800)}. キャッシュは ckpt の更新時刻で無効化."""
    mtime = ckpt.stat().st_mtime
    if cache.exists():
        z = np.load(cache)
        if float(z["ckpt_mtime"]) == mtime and int(z["stride"]) == stride and sorted(set(z["items"][:, 0].tolist())) == sorted(trials):
            print(f"  [cache] {cache}", flush=True)
            return {"items": z["items"], "shape": z["shape"], "anchor": z["anchor"]}
    trainer = load_model(ckpt)
    ds = CenterBeatWindows(trials, stride=stride, labeled_only=False)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=False, num_workers=3)
    shape, anchor, t0 = [], [], time.time()
    for x, _ in loader:
        out = trainer.model(x.to(trainer.device))
        shape.append(out["ECG_shape"].squeeze(1).float().cpu().numpy())
        anchor.append(out["Anchor"].squeeze(1).float().cpu().numpy())
    res = {"items": np.asarray(ds.items, dtype=np.int64), "shape": np.concatenate(shape), "anchor": np.concatenate(anchor)}
    print(f"  推論 {len(ds)} 窓 {time.time() - t0:.0f} s ({ckpt.name})", flush=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, ckpt_mtime=mtime, stride=stride, **res)
    del trainer
    torch.cuda.empty_cache()
    return res


# ---------------------------------------------------------------- 窓ごとの指標
def shape_metrics(pred01: np.ndarray, gt: np.ndarray) -> dict:
    """pred01: 出力([0, 1] 尺度), gt: 正解の片(ECG の単位). 片ごとの RMSE(2 尺度)・PCC・R²."""
    amp = gt.max() - gt.min()
    y01 = (gt - gt.min()) / amp
    res = y01 - pred01
    rmse01 = float(np.sqrt(np.mean(res ** 2)))
    pc = np.corrcoef(pred01, y01)[0, 1] if pred01.std() > 1e-12 else 0.0
    r2 = 1.0 - float(np.sum(res ** 2) / np.sum((y01 - y01.mean()) ** 2))
    pred_n = (pred01 - pred01.min()) / max(pred01.max() - pred01.min(), 1e-12)
    return {"rmse01": rmse01, "rmse_ecg": rmse01 * float(amp), "pcc": float(pc) * 100.0, "r2": r2,
            "mse01": rmse01 ** 2, "norm_mse01": float(np.mean((y01 - pred_n) ** 2))}     # 後ろ 2 つは学習時の val 指標の再現用


def detect(a01: np.ndarray, thr: float) -> np.ndarray:
    return find_peaks(a01, height=thr, distance=MIN_DIST)[0]


def match(pred_pk: np.ndarray, gt_pk: np.ndarray) -> tuple[list[float], int, int]:
    """正解の各 R 波に最も近い候補. 一致の誤差[ms]・missed 数・どの正解にも一致しない候補の数."""
    errs, missed = [], 0
    used = np.zeros(len(pred_pk), bool)
    for g in gt_pk:
        if len(pred_pk) == 0:
            missed += 1
            continue
        j = int(np.argmin(np.abs(pred_pk - g)))
        d = abs(int(pred_pk[j]) - int(g))
        if d <= MATCH:
            errs.append(d * MS_PER_SAMPLE)
            used[j] = True
        else:
            missed += 1
    return errs, missed, int((~used).sum())


# ---------------------------------------------------------------- 集計(試行 → 参加者 → 全体)
def aggregate(per_trial: dict[int, float], by_subject: dict[int, list[int]]) -> tuple[float, float, dict]:
    """試行平均・(参加者平均 → 参加者間平均)・参加者ごとの値."""
    subj = {s: float(np.mean([per_trial[t] for t in ts if t in per_trial])) for s, ts in by_subject.items()
            if any(t in per_trial for t in ts)}
    return float(np.mean(list(per_trial.values()))), float(np.mean(list(subj.values()))), subj


def mean_shape_of(train_trials: list[int]) -> np.ndarray:
    """学習側の対象の片([0, 1] 尺度)の平均. 学習と同じ 0.8 s 刻み・正解のある窓."""
    acc, n = np.zeros(200), 0
    for t in train_trials:
        tr = load_trial(t)
        ecg, peaks = tr["ecg"].astype(np.float32), tr["peaks"].astype(np.int64)
        for s in range(0, len(tr["rcg_z"]) - WINDOW_LEN + 1, rtrain.TRAIN_STRIDE):
            if not np.isfinite(target_rr(peaks, s)):
                continue
            p = gt_piece(ecg, peaks, s)
            if p is not None and p.max() - p.min() > 1e-12:
                acc += minmax(p)
                n += 1
    return acc / max(n, 1)


def evaluate_unit(shape_out: dict, anchor_out: dict, mean_shape: np.ndarray, windows: str, strides: list[int], base: int) -> dict:
    """1 つの run(または 1 分割)の test 試行について, 窓集合 × 刻みごとの試行別の値を返す."""
    trials = sorted(set(shape_out["items"][:, 0].tolist()))
    data = {t: load_trial(t) for t in trials}
    sets = ["labeled"] + (["all"] if windows == "all" else [])
    out = {f"{w}_s{st}": {"shape": {}, "shape_mean": {}, "anchor": {}, "pooled": {}} for w in sets for st in strides}
    diag = {"raw_min": [], "raw_max": [], "at_gt": [], "best_false": [], "raw_at_gt": []}
    for t in trials:
        ecg, peaks = data[t]["ecg"].astype(np.float32), data[t]["peaks"].astype(np.int64)
        idx_s = np.where(shape_out["items"][:, 0] == t)[0]
        idx_a = np.where(anchor_out["items"][:, 0] == t)[0]
        assert np.array_equal(shape_out["items"][idx_s], anchor_out["items"][idx_a])
        rows = []
        for i_s, i_a in zip(idx_s, idx_a):
            s = int(shape_out["items"][i_s, 1])
            labeled = bool(np.isfinite(target_rr(peaks, s)))
            piece = gt_piece(ecg, peaks, s)
            sm = mm = None
            if piece is not None and piece.max() - piece.min() > 1e-12:
                sm = shape_metrics(shape_out["shape"][i_s].astype(np.float64), piece)
                mm = shape_metrics(mean_shape, piece)
            raw = anchor_out["anchor"][i_a].astype(np.float64)
            a01 = minmax(raw)
            gpk = gt_rpeaks(peaks, s)
            per_thr = {}
            for thr in THR_SWEEP:
                e, m, fp = match(detect(a01, thr), gpk)
                per_thr[thr] = (e, m, fp)
            # 値域の診断(しきい値の根拠): 正解の R 波 ±150 ms 内の最大値と, どの正解からも 150 ms 超離れた局所最大の最大値
            if labeled:
                diag["raw_min"].append(raw.min()); diag["raw_max"].append(raw.max())
                near = np.zeros(WINDOW_LEN, bool)
                for g in gpk:
                    near[max(0, g - MATCH):g + MATCH + 1] = True
                    diag["at_gt"].append(a01[max(0, g - MATCH):g + MATCH + 1].max())
                    diag["raw_at_gt"].append(raw[max(0, g - MATCH):g + MATCH + 1].max())
                lm = find_peaks(a01, distance=MIN_DIST)[0]
                lm = lm[~near[lm]]
                diag["best_false"].append(a01[lm].max() if len(lm) else 0.0)
            gt01 = np.zeros(WINDOW_LEN)
            gt01[gpk] = 1.0
            rows.append((s, labeled, sm, mm, per_thr, len(gpk), float(np.mean((a01 - gt01) ** 2))))
        for w in sets:
            for st in strides:
                sel = [r for r in rows if (w == "all" or r[1]) and r[0] % st == 0]
                key = f"{w}_s{st}"
                shp = [r[2] for r in sel if r[2] is not None]
                shm = [r[3] for r in sel if r[3] is not None]
                if shp:
                    out[key]["shape"][t] = {k: float(np.mean([d[k] for d in shp])) for k in shp[0]} | {"n": len(shp)}
                    out[key]["shape_mean"][t] = {k: float(np.mean([d[k] for d in shm])) for k in shm[0]}
                an = {}
                for thr in THR_SWEEP:
                    errs = [e for r in sel for e in r[4][thr][0]]
                    missed = sum(r[4][thr][1] for r in sel)
                    fp = sum(r[4][thr][2] for r in sel)
                    n_gt = sum(r[5] for r in sel)
                    an[thr] = {"timing_mean": float(np.mean(errs)) if errs else float("nan"),
                               "timing_median": float(np.median(errs)) if errs else float("nan"),
                               "mdr": 100.0 * missed / max(n_gt, 1), "fp_per_win": fp / max(len(sel), 1),
                               "n_gt": n_gt, "n_win": len(sel)}
                out[key]["anchor"][t] = an
                out[key].setdefault("pooled", {})[t] = {"n": len(shp), "mse01": float(np.sum([d["mse01"] for d in shp])),
                                                       "norm_mse01": float(np.sum([d["norm_mse01"] for d in shp])),
                                                       "anchor_mse": float(np.sum([r[6] for r in sel])), "n_a": len(sel)}
    return {"per_trial": out, "diag": {k: np.asarray(v, dtype=np.float64) for k, v in diag.items()}}


# ---------------------------------------------------------------- run の構成
def epoch_info(run_dir: Path, task: str, ckpt: Path) -> dict:
    info = {"ckpt": ckpt.name}
    hist = []
    if (run_dir / "metrics.json").exists():
        hist = json.loads((run_dir / "metrics.json").read_text())["history"]
    info["n_epochs_done"] = len(hist)
    if hist:
        info["history_min_epoch"] = int(np.argmin([h["val"]["metrics"][task][0] for h in hist]))
    if ckpt.name == "resume.pt":
        info["epoch"] = len(hist) - 1
        info["selected_by"] = "最終エポック"
    elif ckpt.name == f"best_{task}.pt" and (run_dir / f"best_{task}_epoch.json").exists():
        j = json.loads((run_dir / f"best_{task}_epoch.json").read_text())
        info["epoch"] = j.get("epoch")
        info["history_min_epoch"] = j.get("history_min_epoch", info.get("history_min_epoch"))
    elif (run_dir / "best_epoch.json").exists():
        info["epoch"] = json.loads((run_dir / "best_epoch.json").read_text()).get("epoch")
        info["selected_by"] = "PPI の val"
    return info


def units(run: str) -> list[dict]:
    root = REPO / "runs" / run
    if not root.exists():
        sys.exit(f"{root} がない")
    if (root / "config.json").exists() and json.loads((root / "config.json").read_text())["fold"].get("split") == "official":
        cfg = json.loads((root / "config.json").read_text())
        return [{"name": run, "dir": root, "cfg": cfg, "train": cfg["fold"]["train_trials"], "test": cfg["fold"]["test_trials"]}]
    res = []
    for fold in loso_folds():
        d = root / f"fold{fold.index:02d}_test{fold.test[0]}"
        if not (d / "config.json").exists():
            print(f"[欠け] {d.name}: config.json がない", flush=True)
            continue
        if not any((d / f).exists() for f in ("best_ECG_shape.pt", "best_Anchor.pt", "best_model.pt")):
            print(f"[欠け] {d.name}: チェックポイントがない", flush=True)
            continue
        res.append({"name": d.name, "dir": d, "cfg": json.loads((d / "config.json").read_text()),
                    "train": trials_of(fold.train), "test": trials_of(fold.test), "subject": fold.test[0],
                    "finished": (d / "predictions.npz").exists()})
    return res


def pick_ckpt(d: Path, task: str) -> Path:
    p = d / f"best_{task}.pt"
    return p if p.exists() else d / "best_model.pt"


# ---------------------------------------------------------------- 表
def fmt(x: float, nd: int = 3) -> str:
    return "—" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--windows", choices=["all", "labeled"], default="labeled")
    ap.add_argument("--stride", type=int, default=40)
    ap.add_argument("--thr", type=float, default=0.5, help="アンカー出力(窓ごと min-max 後)のしきい値")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--ckpt", choices=["auto", "last"], default="auto")
    args = ap.parse_args()
    assert args.thr in THR_SWEEP, f"--thr は {THR_SWEEP} のどれか"
    strides = [args.stride] + ([160] if args.stride != 160 and 160 % args.stride == 0 else [])
    by_subject = trials_by_subject()
    us = units(args.run)
    official = len(us) == 1 and us[0]["cfg"]["fold"].get("split") == "official"
    for u in us:
        if (u["cfg"].get("sst"), u["cfg"].get("target")) != (SST_VARIANT, TARGET):
            sys.exit(f"{u['name']}: config の sst/target {u['cfg'].get('sst')}/{u['cfg'].get('target')} と環境変数 {SST_VARIANT}/{TARGET} が食い違う")

    per_trial = {}                      # key -> {"shape": {t: ...}, ...}
    diags, ckinfo = [], []
    for u in us:
        print(f"[{u['name']}] test 試行 {u['test']}", flush=True)
        ck_s, ck_a = ((u["dir"] / "resume.pt",) * 2 if args.ckpt == "last"
                      else (pick_ckpt(u["dir"], "ECG_shape"), pick_ckpt(u["dir"], "Anchor")))
        cdir = REPO / "cache" / "radarode_task_preds" / args.run
        cache = lambda ck: cdir / f"{u['name']}_{ck.stem}_s{args.stride}.npz"  # noqa: E731
        out_s = infer(ck_s, u["test"], args.stride, args.batch, cache(ck_s))
        out_a = out_s if ck_a == ck_s else infer(ck_a, u["test"], args.stride, args.batch, cache(ck_a))
        ms = mean_shape_of(u["train"])
        r = evaluate_unit(out_s, out_a, ms, args.windows, strides, args.stride)
        for key, v in r["per_trial"].items():
            dst = per_trial.setdefault(key, {"shape": {}, "shape_mean": {}, "anchor": {}, "pooled": {}})
            for kk in dst:
                dst[kk].update(v[kk])
        diags.append(r["diag"])
        ckinfo.append({"unit": u["name"], "subject": u.get("subject"), "finished": u.get("finished", True),
                       "ECG_shape": epoch_info(u["dir"], "ECG_shape", ck_s), "Anchor": epoch_info(u["dir"], "Anchor", ck_a)})

    # ---- 集計
    summary = {}
    for key, v in per_trial.items():
        s = {}
        for name, src in (("model", v["shape"]), ("mean_shape", v["shape_mean"])):
            for m in ("rmse01", "rmse_ecg", "pcc", "r2"):
                tm, sm, subj = aggregate({t: d[m] for t, d in src.items()}, by_subject)
                s[f"{name}_{m}"] = {"trial_mean": tm, "subject_mean": sm, "by_subject": subj}
        for thr in THR_SWEEP:
            for m in ("timing_mean", "timing_median", "mdr", "fp_per_win"):
                vals = {t: d[thr][m] for t, d in v["anchor"].items() if np.isfinite(d[thr][m])}
                tm, sm, subj = aggregate(vals, by_subject)
                s[f"anchor_thr{thr}_{m}"] = {"trial_mean": tm, "subject_mean": sm, "by_subject": subj}
        pl = v["pooled"].values()
        s["pooled_sqrt_mse01"] = float(np.sqrt(sum(d["mse01"] for d in pl) / max(sum(d["n"] for d in pl), 1)))
        s["pooled_mse01"] = float(sum(d["mse01"] for d in pl) / max(sum(d["n"] for d in pl), 1))
        s["pooled_norm_mse01"] = float(sum(d["norm_mse01"] for d in pl) / max(sum(d["n"] for d in pl), 1))
        s["pooled_anchor_mse"] = float(sum(d["anchor_mse"] for d in pl) / max(sum(d["n_a"] for d in pl), 1))
        s["n_windows_shape"] = int(sum(d["n"] for d in v["shape"].values()))
        s["n_windows_anchor"] = int(sum(d[THR_SWEEP[0]]["n_win"] for d in v["anchor"].values()))
        s["n_rpeaks"] = int(sum(d[THR_SWEEP[0]]["n_gt"] for d in v["anchor"].values()))
        summary[key] = s
    dg = {k: np.concatenate([d[k] for d in diags]) for k in diags[0]}
    q = lambda x: {p: float(np.percentile(x, p)) for p in (1, 5, 25, 50, 75, 95, 99)}  # noqa: E731
    diag_out = {"raw_min": q(dg["raw_min"]), "raw_max": q(dg["raw_max"]), "raw_at_gt": q(dg["raw_at_gt"]),
                "norm_at_gt": q(dg["at_gt"]), "norm_best_false": q(dg["best_false"]),
                "frac_gt_below": {thr: float(np.mean(dg["at_gt"] < thr)) for thr in THR_SWEEP},
                "frac_false_above": {thr: float(np.mean(dg["best_false"] >= thr)) for thr in THR_SWEEP}}

    agg = "trial_mean" if official else "subject_mean"
    agg_name = "試行平均(11 試行)" if official else "試行 → 参加者 → 参加者間平均"
    split_name = "公開分割(test = 試行 75〜85, 参加者 29・30. val = test でエポックを選んでいる)" if official else "LOSO 11 分割"
    L = [f"# radarODE-MTL 等重み: 波形・アンカーの評価({args.run})", "",
         f"- 分割: {split_name}. 集計: {agg_name}.",
         f"- 入力 RADARODE_SST={SST_VARIANT}, 対象の拍 RADARODE_TARGET={TARGET}. 窓 4 s, 刻み {', '.join(f'{st / FS:.1f} s' for st in strides)}.",
         f"- アンカーのしきい値 {args.thr}(出力を窓ごとに min-max で [0, 1] にした後). 候補の最小間隔 0.3 s. 一致の許容 ±{MATCH_MS:.0f} ms.",
         "- 150 ms は論文の MDR の記述(\"misaligned ECG pieces with small deviations (<150 ms) ... will not be identified as 'missed detected'\")に合わせた. timing error の許容窓は論文に書かれていない.",
         "- 生成: scripts/radarode_task_eval.py", "",
         "## 使ったチェックポイント", "",
         "| 単位 | 学習済みエポック数 | 波形の重み | 選んだエポック | 波形の val 最小エポック | アンカーの重み | 選んだエポック | アンカーの val 最小エポック |",
         "|---|---|---|---|---|---|---|---|"]
    for c in ckinfo:
        e, a = c["ECG_shape"], c["Anchor"]
        L.append(f"| {c['unit']}{'' if c['finished'] else '(学習中)'} | {e.get('n_epochs_done')} | {e['ckpt']}"
                 f"{'(PPI の val 最良で代用)' if e['ckpt'] == 'best_model.pt' else '(最終エポック)' if e['ckpt'] == 'resume.pt' else ''} | {e.get('epoch')} | {e.get('history_min_epoch')} | "
                 f"{a['ckpt']}{'(代用)' if a['ckpt'] == 'best_model.pt' else ''} | {a.get('epoch')} | {a.get('history_min_epoch')} |")
    L += ["", "val 最小エポックは metrics.json の履歴で各タスクの val 指標 0 番(波形 norm_MSE, アンカー MSE)が最小のエポック.", ""]

    if official and len(ckinfo) == 1:
        ep = ckinfo[0]["ECG_shape"].get("epoch")
        hist = json.loads((us[0]["dir"] / "metrics.json").read_text())["history"]
        vm = hist[ep]["val"]["metrics"] if ep is not None and ep < len(hist) else None
        k160 = "labeled_s160"
        if vm is not None and k160 in summary:
            L += ["## 学習時の val 指標の再現(公開分割は val = test. 0.8 s 刻み・正解のある窓・窓をまとめた平均)", "",
                  "| 量 | 学習時の記録(metrics.json, そのエポック) | 本スクリプト |", "|---|---|---|",
                  f"| 波形 MSE([0, 1] 尺度, 出力そのまま) | {vm['ECG_shape'][1]:.5f} | {summary[k160]['pooled_mse01']:.5f} |",
                  f"| 波形 norm_MSE(出力も min-max) | {vm['ECG_shape'][0]:.5f} | {summary[k160]['pooled_norm_mse01']:.5f} |",
                  f"| アンカー MSE(出力を min-max) | {vm['Anchor'][0]:.5f} | {summary[k160]['pooled_anchor_mse']:.5f} |",
                  "", "学習時の記録はバッチごとの平均をバッチの大きさで重みづけしたもの(= 窓をまとめた平均)で, 窓数は同じ 2387.", ""]
    L += ["## 論文 Table I(等重み行)との比較", "",
          "| 指標 | 論文 | 本評価(刻み " + " / ".join(f"{st / FS:.1f} s" for st in strides) + ") | 対照: 学習側の平均の片を出すだけ |", "|---|---|---|---|"]
    keys = [f"labeled_s{st}" for st in strides]
    def row(label, paper, metric, nd, mean_metric=None):
        vals = " / ".join(fmt(summary[k][metric][agg], nd) for k in keys)
        mv = " / ".join(fmt(summary[k][mean_metric][agg], nd) for k in keys) if mean_metric else "—"
        L.append(f"| {label} | {paper} | {vals} | {mv} |")
    row("RMSE([0, 1] 尺度. 学習の損失と同じ)", f"{PAPER['rmse']} mV", "model_rmse01", 3, "mean_shape_rmse01")
    row("RMSE(ECG の単位に戻した値)", f"{PAPER['rmse']} mV", "model_rmse_ecg", 3, "mean_shape_rmse_ecg")
    row("PCC [%]", PAPER["pcc"], "model_pcc", 1, "mean_shape_pcc")
    row("R²", PAPER["r2"], "model_r2", 3, "mean_shape_r2")
    row("Timing error 平均 [ms]", PAPER["timing_ms"], f"anchor_thr{args.thr}_timing_mean", 2)
    row("Timing error 中央値 [ms]", "(集計の記載なし)", f"anchor_thr{args.thr}_timing_median", 2)
    row("MDR [%]", PAPER["mdr"], f"anchor_thr{args.thr}_mdr", 2)
    row("一致しない候補 [個/窓]", "—", f"anchor_thr{args.thr}_fp_per_win", 3)
    L += ["", "窓数: " + ", ".join(f"刻み {k.split('_s')[1]}: 波形 {summary[k]['n_windows_shape']} 窓・アンカー {summary[k]['n_windows_anchor']} 窓・正解の R 波 {summary[k]['n_rpeaks']} 個" for k in keys), ""]
    if args.windows == "all":
        L += ["### 全窓(正解の拍が窓をはみ出す窓も含む)", "", "| 指標 | " + " / ".join(f"刻み {st / FS:.1f} s" for st in strides) + " |", "|---|---|"]
        ka = [f"all_s{st}" for st in strides]
        for label, m, nd in (("RMSE [0, 1]", "model_rmse01", 3), ("RMSE ECG 単位", "model_rmse_ecg", 3), ("PCC [%]", "model_pcc", 1), ("R²", "model_r2", 3),
                             ("Timing 平均 [ms]", f"anchor_thr{args.thr}_timing_mean", 2), ("Timing 中央値 [ms]", f"anchor_thr{args.thr}_timing_median", 2),
                             ("MDR [%]", f"anchor_thr{args.thr}_mdr", 2)):
            L.append(f"| {label} | " + " / ".join(fmt(summary[k][m][agg], nd) for k in ka) + " |")
        L.append("")

    k0 = keys[0]
    L += [f"## 参加者別(正解のある窓, 刻み {strides[0] / FS:.1f} s)", ""]
    subs = list(summary[k0]["model_pcc"]["by_subject"])
    L += ["| 指標 | " + " | ".join(str(s) for s in subs) + " |", "|---|" + "---|" * len(subs)]
    for label, m, nd in (("RMSE [0, 1]", "model_rmse01", 3), ("RMSE ECG 単位", "model_rmse_ecg", 3), ("PCC [%]", "model_pcc", 1),
                         ("R²", "model_r2", 3), ("平均の片の PCC [%]", "mean_shape_pcc", 1),
                         ("Timing 平均 [ms]", f"anchor_thr{args.thr}_timing_mean", 2), ("MDR [%]", f"anchor_thr{args.thr}_mdr", 2)):
        L.append(f"| {label} | " + " | ".join(fmt(summary[k0][m]["by_subject"].get(s), nd) for s in subs) + " |")
    L.append("")

    L += ["## アンカー: しきい値の根拠(出力の値域)", "",
          f"正解のある窓 {len(dg['raw_max'])} 個. 百分位 1 / 5 / 25 / 50 / 75 / 95 / 99:", "",
          "| 量 | 百分位 |", "|---|---|"]
    for label, k in (("生の出力の窓内最小", "raw_min"), ("生の出力の窓内最大", "raw_max"), ("生の出力: 正解の R 波 ±150 ms 内の最大", "raw_at_gt"),
                     ("min-max 後: 正解の R 波 ±150 ms 内の最大", "norm_at_gt"), ("min-max 後: 正解から 150 ms 超離れた局所最大の最大", "norm_best_false")):
        L.append(f"| {label} | " + " / ".join(f"{v:.3f}" for v in diag_out[k].values()) + " |")
    L += ["", "| しきい値 | 正解 ±150 ms 内の最大がしきい値未満の割合 | 正解から離れた山がしきい値以上の窓の割合 | Timing 平均 [ms] | Timing 中央値 [ms] | MDR [%] | 一致しない候補 [個/窓] |",
          "|---|---|---|---|---|---|---|"]
    for thr in THR_SWEEP:
        L.append(f"| {thr}{' (採用)' if thr == args.thr else ''} | {100 * diag_out['frac_gt_below'][thr]:.1f} % | {100 * diag_out['frac_false_above'][thr]:.1f} % | "
                 f"{fmt(summary[k0][f'anchor_thr{thr}_timing_mean'][agg], 2)} | {fmt(summary[k0][f'anchor_thr{thr}_timing_median'][agg], 2)} | "
                 f"{fmt(summary[k0][f'anchor_thr{thr}_mdr'][agg], 2)} | {fmt(summary[k0][f'anchor_thr{thr}_fp_per_win'][agg], 3)} |")
    L.append("")

    res_dir = REPO / "results"
    sfx = "_tasks" + ("_last" if args.ckpt == "last" else "")
    (res_dir / f"{args.run}{sfx}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (res_dir / f"{args.run}{sfx}.json").write_text(json.dumps(
        {"run": args.run, "official": official, "aggregate": agg, "strides": strides, "thr": args.thr, "windows": args.windows,
         "ckpt_mode": args.ckpt, "checkpoints": ckinfo, "summary": summary, "anchor_value_range": diag_out, "paper": PAPER}, indent=1, default=float), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
