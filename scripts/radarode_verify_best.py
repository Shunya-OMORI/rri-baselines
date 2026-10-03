"""各分割の best_model.pt で val を計算し直し, best_epoch.json の記録(PPI_sec)と照合する(2026-10-03 監査)."""
import json, sys, torch
from pathlib import Path
sys.argv = ["x", "0", "200"]
import rri_baselines.radarode.train as T
from rri_baselines.protocol import loso_folds, trials_of
R = T.REPO / "runs/radarode_ew_wsst200"
for f in loso_folds():
    d = R / f"fold{f.index:02d}_test{f.test[0]}"
    rec = json.loads((d / "best_epoch.json").read_text())
    g = torch.Generator(); g.manual_seed(42)
    tr = T.build_trainer(Path("/tmp/omori_verify_best"), 200, g)
    tr.model.load_state_dict(torch.load(d / "best_model.pt", map_location=tr.device))
    vs = T.CenterBeatWindows(trials_of(f.val), T.TRAIN_STRIDE)
    res, _ = tr._evaluate(torch.utils.data.DataLoader(vs, batch_size=32, shuffle=False, num_workers=0))
    a, b = res["PPI"][0], rec["val"]["metrics"]["PPI"][0]
    print(f"{d.name} epoch {rec['epoch']}: 計算し直し {a*1000:.4f} ms / 記録 {b*1000:.4f} ms / 差 {abs(a-b)*1000:.2e} ms", flush=True)
