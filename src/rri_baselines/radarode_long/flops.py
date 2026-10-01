"""radarODE(SCEG + 長期)のパラメータ数と計算量. FLOPpy(UniversalFlopCounter)で数える.

  (a) 4 s 窓 1 つから RRI 1 つ: 窓の中の拍の数だけ SCEG + 長期 1 回(SST・KDE は網の外なので別に書く)
  (b) 順伝播 1 回: SCEG 1 回(1 拍)・長期 1 回(1 窓)
  (c) 出力 1 秒あたり: 4 s の ECG を重ねずに並べるとき (a)/4. 評価と同じ 0.2 s 刻みで窓を出すとき (a)×5
窓の中の拍の数は, 周期 [R − 0.35P, R + 0.65P) が窓と重なる数の期待値 800/P + 1(P = 窓中央の正解 RR, 全 91 試行の評価窓の平均).

  PYTHONPATH=src RADARODE_ROOT=../radarODE-MTL ../radarODE-MTL/.venv/bin/python -m rri_baselines.radarode_long.flops
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from rri_baselines.protocol import FS, WINDOW_LEN, eval_windows, load_trial, trials_by_subject
from rri_baselines.radarode_long.model import LongTerm, build_sceg, count_params

REPO = Path(__file__).resolve().parents[3]


def flops(model: torch.nn.Module, *inputs) -> float:
    from floppy.backends.torch_impl.hooks.universal_flop_counter import UniversalFlopCounter
    model.eval()
    with torch.no_grad(), UniversalFlopCounter() as c:
        model(*inputs)
    return float(c.flops)


def mean_beats_per_window() -> float:
    n = []
    for ts in trials_by_subject().values():
        for t in ts:
            tr = load_trial(t)
            _, rr = eval_windows(len(tr["rcg_z"]), tr["peaks"])
            p = rr[np.isfinite(rr)] / 1000.0 * FS
            n.append(WINDOW_LEN / p + 1)
    return float(np.mean(np.concatenate(n)))


def sst_flops(n: int = WINDOW_LEN, na: int = 80, ch: int = 50) -> float:
    """窓 1 つの SST の概算: FFT 1 回 + 尺度ごとの逆 FFT 2 回(5N log2 N)+ 要素ごとの演算(約 30 回/点)."""
    fft = 5 * n * np.log2(n)
    return ch * (fft + na * 2 * fft + na * n * 30)


def main() -> None:
    out = {}
    nb = mean_beats_per_window()
    lt = LongTerm()
    f_long = flops(lt, torch.randn(1, 50, WINDOW_LEN), torch.randn(1, 1, WINDOW_LEN))
    out["long_term"] = {"params": count_params(lt), "params_encoder": count_params(lt.encoder),
                        "params_decoder": count_params(lt.decoder), "params_tcn": count_params(lt.fusion), "flops": f_long}
    for width in ("table1", "cft"):
        m = build_sceg(width)
        f_sceg = flops(m, torch.rand(1, 50, 71, 120))
        per_window = nb * f_sceg + f_long
        out[f"sceg_{width}"] = {
            "params": count_params(m), "params_backbone": count_params(m.backbone.backbone),
            "params_squeezer": count_params(m.backbone.squeeze_module), "params_encoder": count_params(m.shape.encoder),
            "params_decoder": count_params(m.shape.decoder), "params_ode_estimator": count_params(m.shape.decoder.param_estimator),
            "flops_per_beat": f_sceg,
            "total_params": count_params(m) + count_params(lt),
            "flops_per_window": per_window, "flops_per_output_second_tiled": per_window / 4.0,
            "flops_per_second_at_0.2s_stride": per_window * 5.0}
    out["mean_beats_per_window"] = nb
    out["sst_flops_per_window_estimate"] = sst_flops()
    out["paper_table8"] = {"params_M": 6.04, "flops_G": 1.45, "note": "arXiv v2 Table VIII p.14, 'one forward propagation'"}
    path = REPO / "results/radarode_long/complexity.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
