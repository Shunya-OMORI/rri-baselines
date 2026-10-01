"""wsst の移植: 行の周波数の並びと, 取り出す後半 71 行の帯域(2026-09-30 の帯域の取り違えを再発させない)."""
import numpy as np
import torch

from rri_baselines.radarode.wsst import trial_sst, wsst

N, FS = 35505, 200


def test_rows_ascend_and_upper_half_is_0p72_to_100hz():
    sst, freqs = wsst(torch.zeros(N).normal_(generator=torch.Generator().manual_seed(0)))
    assert sst.shape == (140, N)                                  # floor(log2 N) − 1 = 14 オクターブ × 10 本
    assert np.all(np.diff(freqs) > 0)
    half = freqs[len(freqs) // 2 - 1:]                            # MATLAB の freq_len/2:freq_len
    assert len(half) == 71 and 0.70 < half[0] < 0.75 and abs(half[-1] - 100.0) < 1e-6


def test_sine_lands_on_its_row_and_output_is_flipped():
    t = np.arange(N) / FS
    x = np.sin(2 * np.pi * 20.0 * t)
    sst, freqs = wsst(torch.from_numpy(x))
    row = int(sst.abs().mean(dim=1).argmax())
    assert abs(np.log2(freqs[row] / 20.0)) < 0.15                 # 1 行 = 約 0.1 オクターブ
    out = trial_sst(x[:, None].astype(np.float32), "cpu")[0]      # (71, T30), 行 0 が 100 Hz
    assert out.shape == (71, 5326) and 0.0 <= out.min() and out.max() <= 1.0
    top = int(out.mean(axis=1).argmax())
    assert abs(top - round(np.log2(100.0 / 20.0) / (np.log2(N / 2) / 139))) <= 1   # 100 Hz から数えて 20 Hz の行
