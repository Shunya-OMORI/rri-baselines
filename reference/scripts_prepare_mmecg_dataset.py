"""
生のMMECG .mat データから、radarODE-MTL公式実装 (Projects/radarODE_plus) が
期待するデータセット形式 (obj{ID}_{status}_{ID}_/{sst,ecg,anchor}_seg_i.npy) を生成する。

原著リポジトリにはセグメント化スクリプト自体は同梱されていない（MMECG_to_SST.mは
「1チャンネル全区間のSST」を作るところまでで、4秒窓への切り出し方法は非公開）ため、
以下は README / spectrum_dataset.py / nets/anchor_decoder.py から読み取れる制約を
満たすように独自設計したものである。設計判断は最終レポートに明記する。

各サブジェクトの生データ:
    data.RCG: (N, 50) float64, 200Hz  -- レーダの50点RCG信号
    data.ECG: (N, 1)  float64, 200Hz  -- 参照ECG
    data.id, data.physistatus

出力仕様 (原著コードの制約から逆算):
    sst_seg_i.npy   : (50, 71, 120) float32  -- 50ch, 71 freq bins ([1,25]Hzでは
                      なく、MMECG_to_SST.mの「freq_len/2:freq_len (上半分)」に相当
                      するSSTのfull free bins上半分を採用。詳細はコード内コメント参照)、
                      120 frames (4秒 @ 30Hz)。チャンネルごとにmin-max正規化済み。
    ecg_seg_i.npy   : (L,) float32          -- 1心拍分(R_i -> R_{i+1})の生ECG振幅値。
                      Lはサンプル数(200Hzでの間隔=PPI)で可変長。
    anchor_seg_i.npy: (800,) float32        -- 4秒窓(800 samples @ 200Hz)内の
                      R波位置に立てたガウシアンヒートマップ。
                      anchor_decoder(dummy forward, in=(B,1024,31))の出力長を
                      実際に計算したところ800であることを確認済み
                      (4秒*200Hz=800に一致するため、この解釈を採用した)。

SST近似について:
    原著はMATLABの `wsst` (synchrosqueezed wavelet transform) を使用しているが、
    Pythonに完全に同一な実装はないため、`ssqueezepy.ssq_cwt` (synchrosqueezed CWT,
    morletウェーブレット) で近似する。周波数軸は返された全bin数の上半分を採用し、
    71binになるようリサンプル、時間軸は200Hz->30Hzにリサンプルする。
    **これは原著のwsstと完全に同一ではない近似実装である。**
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import scipy.io as sio
from scipy.signal import find_peaks

import ssqueezepy as ssq

RAW_DATA_DIR = "/net/nfs/home/omori26/radar-hrv-remote/data/raw/mmecg/finalPartialPublicData20221108"
FS = 200  # Hz, both RCG and ECG
WINDOW_SEC = 4.0
WINDOW_LEN = int(WINDOW_SEC * FS)  # 800
SST_TIME_FRAMES = 120  # 30Hz * 4s
SST_FREQ_BINS = 71

MIN_RR_SEC = 0.3
PEAK_HEIGHT_ZSCORE = 2.0


def detect_r_peaks(ecg: np.ndarray, fs: int) -> np.ndarray:
    """z-score閾値+不応期によるシンプルなR波検出（dog-radar-vitals/rpeaks.pyと同種の
    アルゴリズムを、このリポジトリ内で独立に再実装したもの）。

    2026-08-06追記(恒久対応の経緯): このアルゴリズムはT波を第二のR波候補として
    検出したり、逆に真のR波を見逃したりすることがユーザ指摘・実データで確認された
    (`reports/progress_report_2026-08-05_presentation.md`付録参照)。以後の再ラベル付けは
    `detect_r_peaks_neurokit`を使う。この関数自体は過去の再現性のために残す。
    """
    z = (ecg - np.nanmean(ecg)) / (np.nanstd(ecg) + 1e-8)
    peaks, _ = find_peaks(z, height=PEAK_HEIGHT_ZSCORE, distance=int(MIN_RR_SEC * fs))
    return peaks


def detect_r_peaks_neurokit(ecg: np.ndarray, fs: int) -> np.ndarray:
    """外部の検証済みQRS検出アルゴリズム(NeuroKit2)によるR波検出(恒久対応)。

    なぜこちらを採用するか: detect_r_peaks(z-score閾値+find_peaks)は生のECGに
    直接閾値をかけるだけで、QRS波を強調しT波を抑制する帯域通過フィルタ等を
    経由しない。そのため振幅の大きいT波を誤ってR波として検出することがある。
    NeuroKit2のecg_clean()はQRS検出のための標準的な前処理(帯域通過フィルタ等、
    Pan & Tompkins, 1985の系譜)を内蔵しており、本プロジェクトのMMECG全91ファイルで
    検証した結果、R-T混同を疑わせる短い(<350ms)拍間隔が0件だった
    (z-score閾値方式も0件で一見同水準に見えるが、この方式は単に検出数が少ない
    ことで短い間隔を作らないだけであり、後述のようにNeuroKit2の方が総検出数が
    多く見逃しが少ない)。

    対応する先行研究: Makowski et al., 2021, "NeuroKit2: A Python toolbox for
    neurophysiological signal processing," Behavior Research Methods。
    """
    import neurokit2 as nk

    cleaned = nk.ecg_clean(ecg, sampling_rate=fs)
    _, info = nk.ecg_peaks(cleaned, sampling_rate=fs)
    return np.asarray(info["ECG_R_Peaks"], dtype=np.int64)


def build_spike_train(peak_indices: np.ndarray, length: int) -> np.ndarray:
    """0/1のスパイク列を作る(ガウシアンではない)。

    2026-08-27追記: CFT-RFcardi(同著者の後続研究, github.com/ZYY0844/CFT-RFcardi)
    が公開している utils/data_generation.ipynb を読むと、実際のanchor生成は
    `anchor_mask_total[item] = 1` という単純な0/1立てで、ガウシアンではない
    (CFT-RFcardiが配布する実データ data_anchor_*.npy を直接確認した結果と一致、
    S7b/S7b3参照)。このスクリプトは元々 build_heatmap(sigma_ms=10.0) という
    「ガウシアンだろう」という自分の推測でanchorを作っていたが、これは誤りだった。
    --anchor_format spike を指定するとこちらを使う。
    """
    spike = np.zeros(length, dtype=np.float32)
    valid = peak_indices[(peak_indices >= 0) & (peak_indices < length)]
    spike[valid] = 1.0
    return spike


def build_heatmap(peak_indices: np.ndarray, length: int, fs: int, sigma_ms: float = 10.0) -> np.ndarray:
    heatmap = np.zeros(length, dtype=np.float32)
    if len(peak_indices) == 0:
        return heatmap
    sigma_samples = sigma_ms / 1000.0 * fs
    window = max(1, int(sigma_samples * 4))
    t = np.arange(-window, window + 1)
    gaussian = np.exp(-0.5 * (t / sigma_samples) ** 2)
    for idx in peak_indices:
        start = max(0, idx - window)
        end = min(length, idx + window + 1)
        g_start = start - (idx - window)
        g_end = g_start + (end - start)
        heatmap[start:end] = np.maximum(heatmap[start:end], gaussian[g_start:g_end])
    return heatmap


def sst_batch(x: np.ndarray, fs: int) -> np.ndarray:
    """(n_ch, WINDOW_LEN) -> (n_ch, SST_FREQ_BINS, SST_TIME_FRAMES) の正規化済みSST振幅。

    2026-08-07高速化: `ssq.ssq_cwt`は2D入力(複数チャンネルの一括バッチ)を受け付けるが、
    元の実装はチャンネルごとに1回ずつ呼び出していた(50ch/窓で約53秒)。1回のバッチ
    呼び出しにまとめると、チャンネル単位ループと数値的に完全に同一の結果を保ちつつ
    （検証済み: max abs diff = 0.0）、約8.7倍高速化する（内部でFFTベースの計算を
    バッチ化できるため、チャンネルごとの繰り返しオーバーヘッドが消える）。
    """
    x = np.ascontiguousarray(x, dtype=np.float64)  # (n_ch, WINDOW_LEN)
    tx, *_ = ssq.ssq_cwt(x, wavelet="morlet", fs=fs)
    mag = np.abs(np.asarray(tx))  # (n_ch, n_freq, WINDOW_LEN)
    n_freq = mag.shape[1]
    # MMECG_to_SST.m: freq_len/2 : freq_len (上半分 = 高周波数寄り半分)
    half = mag[:, n_freq // 2:, :]  # (n_ch, n_freq/2, WINDOW_LEN)
    n_ch = half.shape[0]

    # 周波数軸を71binにリサンプル
    if half.shape[1] != SST_FREQ_BINS:
        src_idx = np.linspace(0, half.shape[1] - 1, half.shape[1])
        dst_idx = np.linspace(0, half.shape[1] - 1, SST_FREQ_BINS)
        half = np.stack(
            [np.stack([np.interp(dst_idx, src_idx, half[c, :, t]) for t in range(half.shape[2])], axis=1)
             for c in range(n_ch)], axis=0)
    # 時間軸を200Hz -> 30Hz (120 frames) にリサンプル
    src_t = np.linspace(0, half.shape[2] - 1, half.shape[2])
    dst_t = np.linspace(0, half.shape[2] - 1, SST_TIME_FRAMES)
    out = np.stack(
        [np.stack([np.interp(dst_t, src_t, half[c, f, :]) for f in range(half.shape[1])], axis=0)
         for c in range(n_ch)], axis=0)  # (n_ch, SST_FREQ_BINS, SST_TIME_FRAMES)

    # チャンネルごとにmin-max正規化 [0,1]
    mn = out.min(axis=(1, 2), keepdims=True)
    mx = out.max(axis=(1, 2), keepdims=True)
    span = mx - mn
    out = np.where(span < 1e-12, 0.0, (out - mn) / np.where(span < 1e-12, 1.0, span))
    return out.astype(np.float32)


def process_subject(mat_path: str, out_root: str, step: int, max_segments: int | None,
                     folder_id: int, detector: str = "legacy", anchor_format: str = "gaussian") -> int:
    # NOTE: the internal `data.id` field inside the .mat file does NOT uniquely identify
    # a raw file: several raw files (e.g. different physistatus recordings) can share the
    # same internal id. Using it as the folder ID caused directory collisions/overwrites.
    # We therefore use the raw filename number (folder_id, passed in by the caller) as the
    # dataset "subject ID" instead -- this is also what `ID_all`/`ID_test` in main.py index
    # against, and it is guaranteed unique per raw file.
    d = sio.loadmat(mat_path)
    data = d["data"]
    rcg = data["RCG"][0, 0].astype(np.float64)   # (N, 50)
    ecg = data["ECG"][0, 0].astype(np.float64).squeeze(-1)  # (N,)
    status = str(data["physistatus"][0, 0][0]) if data["physistatus"][0, 0].size else "NA"

    n = rcg.shape[0]
    if n < WINDOW_LEN + FS:
        print(f"  [skip] subject {folder_id}: too short ({n} samples)")
        return 0

    r_peaks = detect_r_peaks_neurokit(ecg, FS) if detector == "neurokit" else detect_r_peaks(ecg, FS)
    if len(r_peaks) < 3:
        print(f"  [skip] subject {folder_id}: too few R peaks ({len(r_peaks)})")
        return 0

    out_dir = os.path.join(out_root, f"obj{folder_id}_{status}_{folder_id}_")
    os.makedirs(out_dir, exist_ok=True)

    seg_i = 0
    start = 0
    while start + WINDOW_LEN <= n:
        if max_segments is not None and seg_i >= max_segments:
            break
        end = start + WINDOW_LEN
        center = start + WINDOW_LEN // 2

        # window内のR波（グローバルindex -> window-local index）
        in_window = r_peaks[(r_peaks >= start) & (r_peaks < end)]
        if len(in_window) < 2:
            start += step
            continue
        local_peaks = in_window - start

        # 中心に最も近いピークを基準ビートの開始R波とする
        nearest_i = np.argmin(np.abs(in_window - center))
        # 次のピークが存在しない（配列末尾）場合はスキップ
        if nearest_i >= len(in_window) - 1:
            nearest_i = max(0, nearest_i - 1)
            if nearest_i >= len(in_window) - 1:
                start += step
                continue
        r_i_global = in_window[nearest_i]
        r_ip1_global = in_window[nearest_i + 1]
        cycle_len = r_ip1_global - r_i_global
        # 論文コメントに合わせ、明らかに異常な周期は除外(50-250 samples @200Hz -> 0.25-1.25s)
        if cycle_len < 40 or cycle_len > 300:
            start += step
            continue

        ecg_seg = ecg[r_i_global:r_ip1_global].astype(np.float32)
        if anchor_format == "spike":
            anchor_seg = build_spike_train(local_peaks, WINDOW_LEN)
        else:
            anchor_seg = build_heatmap(local_peaks, WINDOW_LEN, FS, sigma_ms=10.0)

        rcg_window = rcg[start:end, :]  # (800, 50)
        sst_seg = sst_batch(rcg_window.T, FS)  # (50, 800) -> (50, SST_FREQ_BINS, SST_TIME_FRAMES)

        np.save(os.path.join(out_dir, f"sst_seg_{seg_i}.npy"), sst_seg)
        np.save(os.path.join(out_dir, f"ecg_seg_{seg_i}.npy"), ecg_seg)
        np.save(os.path.join(out_dir, f"anchor_seg_{seg_i}.npy"), anchor_seg)

        seg_i += 1
        start += step

    print(f"  subject {folder_id} ({status}): {seg_i} segments -> {out_dir}")
    return seg_i


def _run_subject(sid: int, out_root: str, step: int, max_segments, detector: str = "legacy", anchor_format: str = "gaussian"):
    mat_path = os.path.join(RAW_DATA_DIR, f"{sid}.mat")
    if not os.path.exists(mat_path):
        print(f"  [missing] {mat_path}")
        return 0
    return process_subject(mat_path, out_root, step, max_segments, folder_id=sid, detector=detector, anchor_format=anchor_format)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default="/home/shunya/research/radarODE-MTL/data_prepared")
    ap.add_argument("--subjects", type=str, default=None,
                     help="comma-separated subject IDs, default: all available in RAW_DATA_DIR")
    ap.add_argument("--n_subjects", type=int, default=None, help="limit number of subjects (for time budget)")
    ap.add_argument("--step_sec", type=float, default=2.0, help="sliding window step in seconds")
    ap.add_argument("--max_segments_per_subject", type=int, default=None)
    ap.add_argument("--n_workers", type=int, default=1)
    ap.add_argument("--detector", choices=["legacy", "neurokit"], default="legacy",
                     help="legacy: z-score+find_peaks(元の方式、R/T混同の恐れあり)。"
                          "neurokit: NeuroKit2による恒久対応(2026-08-06)")
    ap.add_argument("--anchor_format", choices=["gaussian", "spike"], default="gaussian",
                     help="gaussian: 自作のsigma_ms=10ガウシアン(旧デフォルト、推測)。"
                          "spike: CFT-RFcardi(github.com/ZYY0844/CFT-RFcardi)のdata_generation.ipynb"
                          "で確認した0/1スパイク形式(実際の上流仕様、2026-08-27導入)")
    args = ap.parse_args()

    os.makedirs(args.out_root, exist_ok=True)
    step = int(args.step_sec * FS)

    if args.subjects is not None:
        ids = [int(x) for x in args.subjects.split(",")]
    else:
        files = sorted(
            (f for f in os.listdir(RAW_DATA_DIR) if f.endswith(".mat")),
            key=lambda f: int(re.sub(r"\D", "", f))
        )
        ids = [int(re.sub(r"\D", "", f)) for f in files]
        if args.n_subjects is not None:
            ids = ids[: args.n_subjects]

    print(f"Processing {len(ids)} subjects, step={args.step_sec}s ({step} samples), "
          f"n_workers={args.n_workers}, detector={args.detector}")
    t0 = time.time()
    total = 0

    if args.n_workers <= 1:
        for sid in ids:
            total += _run_subject(sid, args.out_root, step, args.max_segments_per_subject, args.detector, args.anchor_format)
    else:
        with ProcessPoolExecutor(max_workers=args.n_workers) as ex:
            futs = {ex.submit(_run_subject, sid, args.out_root, step, args.max_segments_per_subject, args.detector, args.anchor_format): sid for sid in ids}
            for fut in as_completed(futs):
                sid = futs[fut]
                try:
                    total += fut.result()
                except Exception as e:
                    print(f"  [error] subject {sid}: {e}")
    print(f"Done. total segments={total}, elapsed={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
