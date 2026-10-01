"""radarODE の第 1 段: PPI 推定(論文 p.5〜6, Algorithm 1, 式 17〜18)を, 評価する 4 s 窓の中だけで行う.

  1. 50 点それぞれのエネルギー波形 x̂_i(sst.py. 帯域の |SST| を周波数方向に足したもの, 200 Hz, [0, 1])
  2. 長さ l_seg の区間に切る(Algorithm 1 の INITIALIZATION). 窓の中央 RR に使うのは窓の中央に置いた区間
  3. 区間ごと・点ごとに "biopeaks"(NeuroKit2 のピーク検出, 式 17)→ 差分を PPI の候補に足す(Line 8〜10)
  4. 50 点の候補全体にガウス核の KDE(式 18, h = n^(-1/5))をかけ, 密度が最大の p を PPI とする(Line 12〜13)
長期モジュールで ECG 断片を並べる位置(位相)も, 同じ区間の検出ピークから出す: 周期 PPI の円周上で平均した位置 φ.
論文は位相の決め方を書いていない(学習・評価とも正解 ECG の心周期を使ったとみられる). 窓の外の心電図は使えないので, ここで補う.

論文で決まらない点(学習なしなので, 候補をすべて計算しておき val で選ぶ):
  - 帯域 BAND: [1, 25] Hz(p.6・p.9 の本文. 本線)/ 0.72〜100 Hz(著者の MMECG_to_SST.m が実際に取る帯域)
  - l_seg: 2 / 3 / 4 s(論文は値を書いていない. 窓が 4 s なので 4 s 以下). l_step は中央の区間が必ず作れる 0.5 s に固定
    (窓の中央 RR に使うのは中央の区間 1 つなので, l_step は値に効かない)
  - "biopeaks": 論文の引用 [39] は NeuroKit2. NeuroKit2 には biopeaks という関数がない. biopeaks(Brammer 2020)由来の
    検出器は ECG の "neurokit"(勾配の閾値)と PPG の "elgendi"(脈波の極大)の 2 つなので両方を候補にする
  - h = n^(-1/5) の単位: 秒("sec", 文字どおり)/ 標本("sample", 5 ms 単位で文字どおり)/ n^(-1/5)·σ("scott", scipy の既定)
  - 比較用に候補の平均("mean", 論文 Fig. 9 の "PPI using Mean")

  python -m rri_baselines.radarode_long.ppi compute [--device cuda|cpu] [--workers N]   → cache/radarode_long/kde/trial<id>.npz
  python -m rri_baselines.radarode_long.ppi evaluate                                  → results/radarode_long/ppi_kde.json
"""
from __future__ import annotations

import argparse
import json
import os
import warnings
from itertools import product
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from rri_baselines.protocol import FS, WINDOW_LEN, eval_windows, load_trial, loso_folds, score, trials_by_subject, trials_of

REPO = Path(__file__).resolve().parents[3]
CACHE = Path(os.environ.get("RADARODE_LONG_KDE", REPO / "cache/radarode_long/kde"))
RESULTS = REPO / "results/radarode_long"
BANDS = ("paper", "author")
LSEGS = (2.0, 3.0, 4.0)
DETECTORS = ("neurokit", "elgendi")
BWS = ("sec", "sample", "scott", "mean")
GRID_S = np.arange(40, 401) / FS                  # PPI の探索範囲 0.2〜2.0 s. 粗い格子は 1 標本(5 ms)刻み, 最大の近くを 1 ms で詰める
FINE_S = np.arange(-4, 5) / 1000.0


def detect(seg: np.ndarray, method: str) -> np.ndarray:
    """式 17: P = biopeaks(x̂). 区間 (L,) の中のピーク位置[標本]."""
    import neurokit2 as nk
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if method == "neurokit":
                p = nk.ecg_findpeaks(seg, sampling_rate=FS, method="neurokit")["ECG_R_Peaks"]
            else:
                p = nk.ppg_findpeaks(seg, sampling_rate=FS, method="elgendi")["PPG_Peaks"]
        return np.asarray(p, dtype=np.int64)
    except Exception:                                # 平らな区間などで落ちるときは検出なし
        return np.zeros(0, dtype=np.int64)


def kde_mode(cand_s: np.ndarray, bw: str) -> float:
    """式 18 の f̂(p) を最大にする p [s]. 候補が無ければ NaN."""
    n = len(cand_s)
    if n == 0:
        return np.nan
    if bw == "mean":
        return float(np.mean(cand_s))
    h = n ** (-1 / 5)
    if bw == "sample":
        h = h / FS
    elif bw == "scott":
        h = h * max(float(np.std(cand_s, ddof=1)) if n > 1 else 0.0, 1.0 / FS)
    def dens(grid):
        return np.exp(-0.5 * ((grid[:, None] - cand_s[None, :]) / h) ** 2).sum(axis=1)
    p0 = GRID_S[int(np.argmax(dens(GRID_S)))]          # 候補は 5 ms の倍数なので, 5 ms 格子は候補の値をすべて含む
    fine = p0 + FINE_S
    return float(fine[int(np.argmax(dens(fine)))])


def circular_phase(peaks: np.ndarray, ppi_samples: float) -> float:
    """検出ピーク(窓の中の位置[標本])を周期 PPI の円周に置いたときの平均の位置 φ ∈ [0, PPI)."""
    if len(peaks) == 0 or not np.isfinite(ppi_samples):
        return np.nan
    z = np.exp(2j * np.pi * peaks / ppi_samples).mean()
    return float((np.angle(z) / (2 * np.pi)) % 1.0 * ppi_samples)


def window_estimates(energy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """1 つの窓・1 つの帯域のエネルギー波形 (50, 800) → PPI[ms]・位相[標本]・候補数. 形は (l_seg, 検出器, h) と (l_seg, 検出器)."""
    ppi = np.full((len(LSEGS), len(DETECTORS), len(BWS)), np.nan, dtype=np.float32)
    phase = np.full_like(ppi, np.nan)
    ncand = np.zeros((len(LSEGS), len(DETECTORS)), dtype=np.int16)
    for i, lseg in enumerate(LSEGS):
        length = int(round(lseg * FS))
        a = (WINDOW_LEN - length) // 2                                                  # 窓の中央に置いた区間
        for j, det in enumerate(DETECTORS):
            peaks = [detect(energy[c, a:a + length].astype(np.float64), det) for c in range(energy.shape[0])]
            cand = np.concatenate([np.diff(p) for p in peaks]).astype(np.float64) / FS
            allp = np.concatenate(peaks) + a
            ncand[i, j] = len(cand)
            for k, bw in enumerate(BWS):
                p_s = kde_mode(cand, bw)
                ppi[i, j, k] = p_s * 1000.0
                phase[i, j, k] = circular_phase(allp, p_s * FS)
    return ppi, phase, ncand


def trial_energies(trial_id: int, device: str) -> tuple[np.ndarray, np.ndarray]:
    """評価の窓すべてのエネルギー波形を 2 つの帯域で. 戻り値 (n, 帯域, 50, 800) float16 と窓の開始位置 (n,)."""
    import torch

    from rri_baselines.radarode_long.sst import WindowSST, _minmax, wsst_batch
    tr = load_trial(trial_id)
    starts, _ = eval_windows(len(tr["rcg_z"]), tr["peaks"])
    rcg = torch.from_numpy(tr["rcg_z"].T.copy())
    ssts = [WindowSST(b, device) for b in BANDS]
    out = np.zeros((len(starts), len(BANDS), rcg.shape[0], WINDOW_LEN), dtype=np.float16)
    chunk = 8 if device != "cpu" else 2
    with torch.no_grad():
        for i in range(0, len(starts), chunk):
            s = starts[i:i + chunk]
            x = torch.stack([rcg[:, a:a + WINDOW_LEN] for a in s]).to(device)
            mag = wsst_batch(x)                                                          # 1 回の SST から 2 つの帯域を取る
            for b, sst in enumerate(ssts):
                out[i:i + len(s), b] = _minmax(mag[:, :, sst.rows].sum(dim=2), dims=(2,)).cpu().numpy()
    return out, starts


def _estimate_windows(energies: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(n, 帯域, 50, 800) → PPI[ms]・位相 (n, 帯域, l_seg, 検出器, h) と候補数 (n, 帯域, l_seg, 検出器)."""
    n = len(energies)
    ppi = np.full((n, len(BANDS), len(LSEGS), len(DETECTORS), len(BWS)), np.nan, dtype=np.float32)
    phase = np.full_like(ppi, np.nan)
    ncand = np.zeros((n, len(BANDS), len(LSEGS), len(DETECTORS)), dtype=np.int16)
    for w in range(n):
        for b in range(len(BANDS)):
            ppi[w, b], phase[w, b], ncand[w, b] = window_estimates(energies[w, b].astype(np.float32))
    return ppi, phase, ncand


def _save(trial_id: int, ppi, phase, ncand, starts) -> str:
    path = CACHE / f"trial{trial_id}.npz"
    tmp = CACHE / f"trial{trial_id}.tmp.npz"
    np.savez(tmp, ppi_ms=ppi, phase=phase, ncand=ncand, starts=starts)
    tmp.rename(path)
    return f"trial {trial_id}: {len(starts)} 窓"


def _detect_trial(args) -> str:
    trial_id, energies, starts = args
    return _save(trial_id, *_estimate_windows(energies), starts)


def _cpu_trial(trial_id: int) -> str:
    """CPU だけで 1 試行: 窓 CHUNK 個ずつ SST → 検出(エネルギー波形を試行ぶん溜めないのでメモリが小さい)."""
    import torch

    from rri_baselines.radarode_long.sst import WindowSST, _minmax, wsst_batch
    torch.set_num_threads(int(os.environ.get("KDE_TORCH_THREADS", "1")))
    tr = load_trial(trial_id)
    starts, _ = eval_windows(len(tr["rcg_z"]), tr["peaks"])
    rcg = torch.from_numpy(tr["rcg_z"].T.copy())
    ssts = [WindowSST(b) for b in BANDS]
    parts = []
    with torch.no_grad():
        for i in range(0, len(starts), 2):
            x = torch.stack([rcg[:, a:a + WINDOW_LEN] for a in starts[i:i + 2]])
            mag = wsst_batch(x)
            en = np.stack([_minmax(mag[:, :, w.rows].sum(dim=2), dims=(2,)).numpy() for w in ssts], axis=1)
            parts.append(_estimate_windows(en))
    return _save(trial_id, *(np.concatenate(z) for z in zip(*parts)), starts)


def compute(device: str, workers: int, trials: list[int] | None = None) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    ids = trials or [t for ts in trials_by_subject().values() for t in ts]
    ids = [t for t in ids if not (CACHE / f"trial{t}.npz").exists()]
    with Pool(workers) as pool:
        if device == "cpu":
            for line in pool.imap_unordered(_cpu_trial, ids):
                print(line, flush=True)
        else:                                          # SST は装置の上で順に, 検出(NeuroKit2)は CPU の作業者に渡す
            pending = []
            for t in ids:
                energies, starts = trial_energies(t, device)
                pending.append(pool.apply_async(_detect_trial, ((t, energies, starts),)))
                while len(pending) > workers:          # メモリを抑える: 渡した試行が作業者数を超えたら待つ
                    print(pending.pop(0).get(), flush=True)
            for p in pending:
                print(p.get(), flush=True)


# ---------------------------------------------------------------- 採点 ----------------------------------------------------------------
def load_cache(trial_id: int) -> dict:
    with np.load(CACHE / f"trial{trial_id}.npz") as z:
        return {k: z[k] for k in z.files}


def config_names() -> list[tuple]:
    return list(product(range(len(BANDS)), range(len(LSEGS)), range(len(DETECTORS)), range(len(BWS))))


def label(cfg: tuple) -> str:
    b, i, j, k = cfg
    return f"{BANDS[b]}/lseg{LSEGS[i]:g}/{DETECTORS[j]}/{BWS[k]}"


def parse_label(name: str) -> tuple:
    b, i, j, k = name.split("/")
    return BANDS.index(b), LSEGS.index(float(i[4:])), DETECTORS.index(j), BWS.index(k)


def _subject_error(trial_ids: list[int], cfg: tuple, fallback_ms: float, cache: dict) -> float:
    errs = []
    for t in trial_ids:
        z, true_ms = cache[t]
        pred = z["ppi_ms"][(slice(None),) + cfg]
        pred = np.where(np.isfinite(pred), pred, fallback_ms)
        ok = np.isfinite(true_ms)
        errs.append(np.mean(np.abs(pred[ok] - true_ms[ok])))
    return float(np.mean(errs))


def anchor_check(fold, cfg: tuple, cache: dict, peaks: dict) -> dict:
    """長期モジュールの形態参照で断片を置く位置の確かさ(学習なし). δ は学習側の試行(窓 4 つに 1 つ)の中央値."""
    from rri_baselines.radarode_long.data import anchors, mechanical_delay
    d = []
    for t in trials_of(fold.train):
        z = cache[t][0]
        sl = slice(None, None, 4)
        d.append(mechanical_delay(z["phase"][(sl,) + cfg], z["ppi_ms"][(sl,) + cfg], z["starts"][sl], peaks[t]))
    delta = float(np.median(np.concatenate(d)))
    err = []
    for t in trials_of(fold.test):
        z = cache[t][0]
        for p, ph, s in zip(z["ppi_ms"][(slice(None),) + cfg], z["phase"][(slice(None),) + cfg], z["starts"]):
            r_hat = anchors(p, ph, delta)
            r = peaks[t][(peaks[t] >= s) & (peaks[t] < s + WINDOW_LEN)] - s
            if len(r_hat) and len(r):
                err.append(np.min(np.abs(r[:, None] - r_hat[None]), axis=1))
    err = np.concatenate(err) / FS * 1000.0
    return {"delta_ms": delta / FS * 1000.0, "anchor_err_median_ms": float(np.median(err)),
            "anchor_within_50ms": float(np.mean(err <= 50))}


def evaluate() -> dict:
    from rri_baselines.ecg_rri import train_mean_rr
    cache, peaks = {}, {}
    for t in [t for ts in trials_by_subject().values() for t in ts]:
        tr = load_trial(t)
        cache[t] = (load_cache(t), eval_windows(len(tr["rcg_z"]), tr["peaks"])[1])
        peaks[t] = tr["peaks"]
    cfgs = config_names()
    by_subject = trials_by_subject()
    folds = loso_folds()
    preds_val, preds_ten, fixed = {}, {}, {c: {} for c in cfgs}
    chosen = []
    for fold in folds:
        fb = train_mean_rr(trials_of(fold.train))
        val_err = {c: np.mean([_subject_error(by_subject[s], c, fb, cache) for s in fold.val]) for c in cfgs}
        ten = [s for s in by_subject if s not in fold.test]                       # 参考: test 以外の 10 名で選ぶ(EQ-Radio と同じ流儀)
        ten_err = {c: np.mean([_subject_error(by_subject[s], c, fb, cache) for s in ten]) for c in cfgs}
        best_val, best_ten = min(cfgs, key=val_err.get), min(cfgs, key=ten_err.get)
        chosen.append({"fold": fold.index, "test": fold.test[0], "val": fold.val[0], "by_val": label(best_val),
                       "val_mae_ms": val_err[best_val], "by_ten": label(best_ten), "ten_mae_ms": ten_err[best_ten]})
        chosen[-1].update(anchor_check(fold, best_val, cache, peaks))
        for t in trials_of(fold.test):
            z, _ = cache[t]
            for c in cfgs:
                p = z["ppi_ms"][(slice(None),) + c]
                fixed[c][t] = np.where(np.isfinite(p), p, fb)
            preds_val[t], preds_ten[t] = fixed[best_val][t], fixed[best_ten][t]
    res_val, res_ten = score(preds_val), score(preds_ten)
    table = sorted(((label(c), score(fixed[c])["mae_ms"]) for c in cfgs), key=lambda kv: kv[1])
    miss = {label(c): float(np.mean(np.concatenate([~np.isfinite(cache[t][0]["ppi_ms"][(slice(None),) + c]) for t in cache])))
            for c in cfgs}
    out = {"mae_ms_by_val": res_val["mae_ms"], "subject_mae_ms_by_val": res_val["subject_mae_ms"],
           "mae_ms_by_ten": res_ten["mae_ms"], "subject_mae_ms_by_ten": res_ten["subject_mae_ms"],
           "chosen": chosen, "fixed_config_mae_ms": dict(table), "no_candidate_share": miss}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "ppi_kde.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    run_root = REPO / "runs/radarode_long_kde"
    for fold in folds:
        d = run_root / f"fold{fold.index:02d}_test{fold.test[0]}"
        d.mkdir(parents=True, exist_ok=True)
        # pred_ms: test 以外の 10 名で選んだ設定(採用), pred_val_ms: val の 1 名で選んだ設定
        np.savez(d / "predictions.npz", **{f"trial{t}_pred_ms": preds_ten[t] for t in trials_of(fold.test)},
                 **{f"trial{t}_pred_val_ms": preds_val[t] for t in trials_of(fold.test)})
    print(f"KDE の PPI(窓中央の RR として, val で選ぶ): {res_val['mae_ms']:.2f} ms / test 以外の 10 名で選ぶ: {res_ten['mae_ms']:.2f} ms")
    for c in chosen:
        print(f"  fold {c['fold']:2d} test {c['test']:2d}: {c['by_val']} (val {c['val_mae_ms']:.1f} ms)")
    print("位相から置いた R 波の位置の誤差(test, 窓の中の正解 R 波ごとに最も近い R̂ との差):")
    for c in chosen:
        print(f"  fold {c['fold']:2d}: δ {c['delta_ms']:.0f} ms, 中央値 {c['anchor_err_median_ms']:.0f} ms, "
              f"50 ms 以内 {100 * c['anchor_within_50ms']:.0f} %")
    print("固定の設定(11 名すべてに同じ設定, 上位 10):")
    for name, v in table[:10]:
        print(f"  {name}: {v:.2f} ms")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["compute", "evaluate"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--trials", type=int, nargs="*")
    a = ap.parse_args()
    if a.cmd == "compute":
        compute(a.device, a.workers, a.trials)
    else:
        evaluate()


if __name__ == "__main__":
    main()
