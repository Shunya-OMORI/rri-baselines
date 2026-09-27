"""R3: 採点の実装が提案手法の既知の値を再現すること."""
from pathlib import Path

import numpy as np
import pytest

from rri_baselines.protocol import loso_folds, score, trials_of

MMWAVE_RUNS = Path("/home/shunya/research/mmwave-rri/runs")


@pytest.mark.skipif(not (MMWAVE_RUNS / "fold00_test1/predictions.npz").exists(), reason="提案手法の予測が無い")
def test_reproduces_mmwave_rri_91131():
    # 公開実装の 91131 相当(2026-09-27 の 11 分割): evaluate_folds の参加者平均 7.019 ms
    preds = {}
    for fold in loso_folds():
        z = np.load(MMWAVE_RUNS / f"fold{fold.index:02d}_test{fold.test[0]}/predictions.npz")
        for t in trials_of(fold.test):
            preds[t] = z[f"trial{t}_pred_ms"]
    assert score(preds)["mae_ms"] == pytest.approx(7.019, abs=5e-4)
