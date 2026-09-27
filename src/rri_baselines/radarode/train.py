"""radarODE-MTL(等重み EW, 3 タスク, 幅 cft, ODE あり)を, 提案手法と同じ分割・目標量・採点で学習・評価する.

モデル・損失・学習器は radarODE-MTL リポジトリの repro/(公開実装の再現レイヤ)をそのまま使い, データだけ data.py に替える.
最適化は repro と同じ(SGD, lr 5e-3, momentum 0.937, weight decay 5e-4, バッチ 32, コサイン). エポック数は EPOCHS
(論文は 200. 計算時間のため縮め, コサインの周期もそれに合わせる). エポックは val の参加者の PPI 誤差で選ぶ(repro と同じ).
test は 0.2 s 間隔の全窓で PPI デコーダの最大のクラス k から RR = (k+1) 標本(repro の損失と整合する読み方)を出す.

  RADARODE_ROOT=<radarODE-MTL> python -m rri_baselines.radarode.train <分割> [エポック数]
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(os.environ.get("RADARODE_ROOT", "/home/shunya/research/radarODE-MTL"))
for p in (ROOT, ROOT / "Projects" / "radarODE_plus"):
    sys.path.insert(0, str(p))

from LibMTL.config import LibMTL_args, prepare_args  # noqa: E402
from LibMTL.utils import set_device, set_random_seed  # noqa: E402
from torch import nn  # noqa: E402

from repro.main import build_task_dict  # noqa: E402
from repro.model import WIDTH_PRESETS, Backbone, ShapeDecoder, build_anchor_decoder, build_ppi_decoder  # noqa: E402
from repro.trainer import ResumableTrainer  # noqa: E402
from rri_baselines.protocol import FS, loso_folds, trials_of  # noqa: E402
from rri_baselines.radarode.data import CenterBeatWindows  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
RUNS = REPO / "runs/radarode_ew"
TRAIN_STRIDE = int(0.8 * FS)          # data_prepared_cft と同じ 0.8 s
TASKS = ["ECG_shape", "PPI", "Anchor"]


def build_trainer(run_dir: Path, epochs: int, loader_gen: torch.Generator):
    params = LibMTL_args.parse_args([])
    params.weighting, params.arch, params.optim, params.scheduler = "EW", "HPS", "sgd", "cos"
    params.lr, params.weight_decay, params.momentum, params.seed = 5e-3, 5e-4, 0.937, 42
    params.eta_min, params.T_max, params.mode = params.lr * 0.01, epochs, "train"
    params.multi_input, params.rep_grad = False, False
    params.save_path, params.load_path = str(run_dir), None
    kwargs, optim_param, scheduler_param = prepare_args(params)
    set_device(params.gpu_id)
    set_random_seed(42)
    widths = WIDTH_PRESETS["cft"]
    decoders = nn.ModuleDict({"ECG_shape": ShapeDecoder(dim=widths[-1], use_ode=True),
                              "PPI": build_ppi_decoder(dim=widths[-1], output_dim=260, variant="cft"),
                              "Anchor": build_anchor_decoder(dim=widths[-1], variant="cft")})
    return ResumableTrainer(
        task_dict=build_task_dict(TASKS), weighting="EW", architecture="HPS",
        encoder_class=lambda: Backbone(in_channels=50, widths=widths), decoders=decoders, rep_grad=False,
        multi_input=False, optim_param=optim_param, scheduler_param=scheduler_param,
        save_path=str(run_dir), load_path=None, modelName="radarode_ew",
        run_dir=str(run_dir), select_task="PPI", select_metric_index=0, select_lower_is_better=True,
        loader_gen=loader_gen, **kwargs)


@torch.no_grad()
def predict(trainer, trial_id: int, batch: int = 64) -> np.ndarray:
    ds = CenterBeatWindows([trial_id], stride=40, labeled_only=False)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=False, num_workers=4)
    trainer.model.eval()
    out = []
    for x, _ in loader:
        logits = trainer.model(x.to(trainer.device))["PPI"]
        out.append((logits.squeeze(1).argmax(dim=1) + 1).float().cpu().numpy() / FS * 1000.0)
    return np.concatenate(out)


def main() -> None:
    index = int(sys.argv[1])
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    fold = loso_folds()[index]
    run_dir = RUNS / f"fold{index:02d}_test{fold.test[0]}"
    if (run_dir / "predictions.npz").exists():
        print(f"[skip] {run_dir.name}", flush=True)
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        (run_dir / "claim").mkdir()              # 同じ分割を 2 つの作業者が回さないための札(mkdir は排他的)
    except FileExistsError:
        print(f"[skip] {run_dir.name}: 他の作業者が学習中", flush=True)
        return
    loader_gen = torch.Generator()
    loader_gen.manual_seed(42)
    train_set = CenterBeatWindows(trials_of(fold.train), TRAIN_STRIDE)
    val_set = CenterBeatWindows(trials_of(fold.val), TRAIN_STRIDE)
    mk = lambda ds, sh, g=None: torch.utils.data.DataLoader(  # noqa: E731
        ds, batch_size=32, shuffle=sh, num_workers=6, pin_memory=True, drop_last=sh, generator=g)
    trainer = build_trainer(run_dir, epochs, loader_gen)
    n_params = sum(p.numel() for p in trainer.model.parameters())
    (run_dir / "config.json").write_text(json.dumps({"fold": fold._asdict(), "weighting": "EW", "width": "cft", "ode": "on",
                                                     "epochs": epochs, "train_stride_samples": TRAIN_STRIDE,
                                                     "n_train": len(train_set), "n_val": len(val_set),
                                                     "n_params": n_params}, indent=1))
    print(f"[{run_dir.name}] train {len(train_set)} / val {len(val_set)} 窓, {n_params:,} パラメータ, {epochs} エポック", flush=True)
    t0 = time.time()
    # test は毎エポック評価しない(0.2 s 間隔の全窓は学習後に 1 回だけ). 学習器の履歴用には val を渡す
    trainer.train(mk(train_set, True, loader_gen), mk(val_set, False), mk(val_set, False), epochs)
    print(f"[{run_dir.name}] 学習 {time.time() - t0:.0f} s", flush=True)
    trainer.model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location=trainer.device))
    preds = {t: predict(trainer, t) for t in trials_of(fold.test)}
    np.savez(run_dir / "predictions.npz", **{f"trial{t}_pred_ms": v for t, v in preds.items()})


if __name__ == "__main__":
    main()
