"""MATLAB で保存した照合用 .mat(matlab_dump.m の出力)を, 我々の移植・参照実装と比べる. MATLAB が使えるようになったとき用.
  python compare_with_matlab_mat.py wsst_check_trial1_p1.mat
"""
import json
import sys

import numpy as np
import scipy.io as sio
import torch

sys.path.insert(0, "/home/shunya/research/rri-baselines/src")
from rri_baselines.radarode.wsst import trial_sst, wsst as wsst_ours  # noqa: E402
from scipy.signal import firwin  # noqa: E402

from compare import diff, windows  # noqa: E402

m = sio.loadmat(sys.argv[1], squeeze_me=True)
x = np.asarray(m["x"], dtype=np.float64)
R = {"matlab_version": str(m.get("ver", "")), "rows": int(m["sst"].shape[0])}
ours, f = wsst_ours(torch.from_numpy(x))
ours = ours.numpy()
R["f"] = diff(f, m["f"]) if m["f"].size == f.size else "行数が違う(R2024a 以降の可能性)"
if m["sst"].shape == ours.shape:
    dd = np.abs(ours / (2 * np.log(2)) - m["sst"])
    R["sst_complex_scaled"] = {"max_abs": float(dd.max()), "rel_to_ref_max": float(dd.max() / np.abs(m["sst"]).max())}
    R["sst_abs"] = diff(np.abs(ours) / (2 * np.log(2)), np.abs(m["sst"]))
h = firwin(401, 1 / 20, window=("kaiser", 5.0)) * 3
R["resample_filter"] = diff(h, m["b"]) if m["b"].size == 401 else f"長さ {m['b'].size}"
final = trial_sst(x[:, None], "cpu")[0].astype(np.float64)
S = np.asarray(m["S"], dtype=np.float64)[::-1]                          # MATLAB は低い方から. load_sst の flip に合わせる
R["final_trial"] = diff(final, S)
R["final_windows"] = windows(final, S)
print(json.dumps(R, indent=1, ensure_ascii=False, default=str))
