"""MMECG の試行を, 公式 block/data_load.py の npyLoad / npyLoadWithHR が読む npy に書き出す.

公式の形式(np.load(path, allow_pickle=True).item() で読む dict):
  rcg (N, 16, 50, 32) float32 … 512 標本 × 50 点を 32 標本のパッチ 16 個に. 公式 block_mask.create_patch_batch で
                                (N, 16, 32, 50) を作り, 点の軸を前に出す. mmJEPA.py は reorder_rcg を通さず
                                B, L, V, _ = rcg.shape と読むので (L, V, P) の並びで置く(end2end/inference の reorder_rcg はどちらも受ける)
  ecg (N, 16, 32, 1) float32  … 同じく create_patch_batch の形. ecg_patches_to_full が (N, 512, 1) に戻す
  pos (N, 50, 3) float32      … MMECG の posxyz(50 点の 3 次元座標). context_encoder の space_proj の入力
  y   (N,) int64              … 公式は姿勢などのラベル(学習では使わない, 長さにだけ使う). ここでは試行番号
  hr  (N,) float32            … 擬似心拍数[bpm](npyLoadWithHR, 事前学習だけ). 書き直し版 mmjepa.data.trial_hr_labels と同じ値
  (trial, start は追跡用. 公式の読み込みは無視する)

正規化は書き直し版で決めたものを引き継ぐ(論文・公式に記載なし):
  レーダ: 512 標本ごと・点ごとの z スコア(窓の外の統計を使わない. 公式の npy 名 "UD_RMS" から窓ごとの正規化と読む)
  ECG   : 512 標本ごとに最小・最大で [-1, 1](DDIM の clip_denoised=True が [-1, 1] を前提にする)

書き出しは pickle の protocol 5 で行う(np.save は protocol 4 で配列を 1 度 bytes に写すので書き出し・読み込みの
ピークが倍になる. protocol 5 は同じ npy として np.load で読め, 配列を写さずに作る).
"""
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch

from rri_baselines.mmjepa.data import HR_MIN_DIST, HR_PROM, TRAIN_STEP, trial_hr_labels
from rri_baselines.mmjepa_official.common import official_on_path
from rri_baselines.protocol import load_trial

official_on_path()
from block.block_mask import create_patch_batch  # noqa: E402  公式の関数で区切る

SEG, PATCH = 512, 32
N_PATCH = 1 + (SEG - PATCH) // PATCH     # 16(公式 num_patch = 1 + (512 - patch_len) // stride)
N_VARS = 50


def norm_rcg(x: np.ndarray) -> np.ndarray:
    """(n, 512, 50) → 窓ごと・点ごとの z スコア. 書き直し版 run.seg_norm と同じ式(torch float32, 不偏標準偏差 + 1e-6)."""
    t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
    t = (t - t.mean(1, keepdim=True)) / (t.std(1, keepdim=True) + 1e-6)
    return t.numpy()


def norm_ecg(e: np.ndarray) -> np.ndarray:
    """(n, 512) → [-1, 1]. 書き直し版 data.minmax と同じ式."""
    t = torch.from_numpy(np.ascontiguousarray(e, dtype=np.float32))
    lo, hi = t.min(1, keepdim=True).values, t.max(1, keepdim=True).values
    return ((t - lo) / (hi - lo).clamp_min(1e-6) * 2 - 1).numpy()


def segments(trial: dict, seg_starts: np.ndarray, chunk: int = 512):
    """試行の中の 512 標本の区間(開始位置 seg_starts)を公式の形で返す (rcg, ecg)."""
    rcg_z, ecg = trial["rcg_z"].astype(np.float32), trial["ecg"].astype(np.float32)
    ar = np.arange(SEG)
    for i in range(0, len(seg_starts), chunk):
        g = seg_starts[i:i + chunk, None] + ar
        r = norm_rcg(rcg_z[g])                                              # (n, 512, 50)
        r = create_patch_batch(r, PATCH, PATCH).transpose(0, 1, 3, 2)        # (n, 16, 32, 50) → (n, 16, 50, 32)
        e = create_patch_batch(norm_ecg(ecg[g])[..., None], PATCH, PATCH)   # (n, 16, 32, 1)
        yield i, r, e


def build(items: list[tuple], with_hr: bool) -> dict:
    """items = [(試行番号, 区間の開始位置 (n,), 擬似心拍数 (n,) か None), ...] → 公式の dict.
    出力の配列を先に確保して試行ごとに埋める(ピークのメモリ ≒ 出力の大きさ)."""
    n = sum(len(s) for _, s, _ in items)
    out = {"rcg": np.empty((n, N_PATCH, N_VARS, PATCH), np.float32),
           "ecg": np.empty((n, N_PATCH, PATCH, 1), np.float32),
           "pos": np.empty((n, N_VARS, 3), np.float32),
           "y": np.empty(n, np.int64), "trial": np.empty(n, np.int64), "start": np.empty(n, np.int64)}
    if with_hr:
        out["hr"] = np.empty(n, np.float32)
    k = 0
    for t, st, hr in items:
        tr = load_trial(t)
        for i, r, e in segments(tr, st):
            out["rcg"][k + i:k + i + len(r)] = r
            out["ecg"][k + i:k + i + len(r)] = e
        out["pos"][k:k + len(st)] = tr["posxyz"].astype(np.float32)[None]
        out["y"][k:k + len(st)] = t
        out["trial"][k:k + len(st)] = t
        out["start"][k:k + len(st)] = st
        if with_hr:
            out["hr"][k:k + len(st)] = hr
        k += len(st)
    return out


def save_npy_dict(path: Path, d: dict) -> int:
    """np.load(path, allow_pickle=True).item() で読める npy(0 次元の object 配列)を protocol 5 で書く."""
    arr = np.empty((), dtype=object)
    arr[()] = d
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as f:
        np.lib.format.write_array_header_2_0(f, np.lib.format.header_data_from_array_1_0(arr))
        pickle.dump(arr, f, protocol=5)
    tmp.rename(path)
    return path.stat().st_size


def train_starts(n_samples: int, step: int = TRAIN_STEP) -> np.ndarray:
    """学習・val の 512 標本の窓(ずらし幅 128 標本. 書き直し版と同じ. 論文に記載なし)."""
    return np.arange(0, n_samples - SEG + 1, step)


def export_split(trials: list[int], path: Path, limit: int | None = None, hr_dist: int = HR_MIN_DIST,
                 hr_prom: float = HR_PROM) -> dict:
    """学習または val の参加者の試行を 1 つの npy に(事前学習の hr 付き. end2end_train.py も同じファイルを読む)."""
    t0 = time.time()
    items = []
    for t in trials:
        st = train_starts(len(load_trial(t)["rcg_z"]))
        items.append((t, st, trial_hr_labels(t, st, hr_dist, hr_prom)))   # 擬似心拍数は試行全体の窓で計算. 検出の細部は分割の学習側で選ぶ
    if limit is not None:            # 動作確認用: 全試行から等間隔に limit 窓
        flat = [(k, j) for k, (_, st, _) in enumerate(items) for j in range(len(st))]
        pick = sorted(set(np.linspace(0, len(flat) - 1, limit).round().astype(int).tolist()))
        by = {}
        for p in pick:
            by.setdefault(flat[p][0], []).append(flat[p][1])
        items = [(items[k][0], items[k][1][js], items[k][2][js]) for k, js in by.items()]
    d = build(items, with_hr=True)
    size = save_npy_dict(path, d)
    info = {"path": str(path), "bytes": size, "n": int(len(d["y"])), "trials": trials, "sec": time.time() - t0,
            "step": TRAIN_STEP, "hr_dist": hr_dist, "hr_prom": hr_prom, "hr_mean": float(d["hr"].mean()), "hr_std": float(d["hr"].std())}
    path.with_suffix(".json").write_text(json.dumps(info, indent=1))
    return info


def export_infer(rows: list[tuple[int, int, int]], out_dir: Path, max_rows: int = 4096) -> list[dict]:
    """推論の入力. rows = [(試行番号, 4 s 窓の番号, 512 標本の区間の開始位置), ...](試行ごとにまとめて並べる).
    max_rows 行ごとに npy を分け(1 ファイル ≒ 420 MB), 行の対応(試行, 窓, 開始位置)を index として返す."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = np.asarray(rows, dtype=np.int64).reshape(-1, 3)
    chunks = []
    # 行数を均等に割る: 公式 inference_from_train.py は最初のバッチの 7 本を図にする(samples[:7].reshape(7, -1))ので,
    # 1 ファイルが 7 行未満だと落ちる. 端数だけの小さなファイルを作らない
    n_chunks = max(1, -(-len(rows) // max_rows))
    bounds = np.linspace(0, len(rows), n_chunks + 1).round().astype(int)
    assert len(rows) >= 7, "公式の推論は 1 回に 7 区間以上が要る"
    for c, (i, j) in enumerate(zip(bounds[:-1], bounds[1:])):
        part = rows[i:j]
        items = []
        for t in dict.fromkeys(part[:, 0].tolist()):
            items.append((t, part[part[:, 0] == t, 2], None))
        d = build(items, with_hr=False)
        d["ecg"][:] = 0.0          # 正解 ECG は渡さない. 公式の推論は生成の後の MSE とログにしか使わない(inference_from_train.py
                                   # の batch ループ)が, 疑いを残さないためゼロで埋める(2026-10-02 レビュー)
        assert np.array_equal(d["start"], np.concatenate([s for _, s, _ in items]))
        path = out_dir / f"chunk{c:03d}.npy"
        size = save_npy_dict(path, d)
        order = np.concatenate([part[part[:, 0] == t] for t, _, _ in items])
        chunks.append({"path": str(path), "bytes": size, "rows": order})
    return chunks
