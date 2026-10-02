"""radarODE-MTL(等重み EW, 3 タスク, 幅 cft, ODE あり)を, 提案手法と同じ分割・目標量・採点で学習・評価する.

モデル・損失・学習器は radarODE-MTL リポジトリの repro/(公開実装の再現レイヤ)をそのまま使い, データだけ data.py に替える.
最適化は repro と同じ(SGD, lr 5e-3, momentum 0.937, weight decay 5e-4, バッチ 32, コサイン). エポック数は EPOCHS
(論文は 200. 20 のときは runs/radarode_ew, それ以外は runs/radarode_ew<エポック数>. コサインの周期もエポック数に合わせる). エポックは val の参加者の PPI 誤差で選ぶ(repro と同じ).
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

from repro.fast_ops import apply as apply_fast_ops  # noqa: E402
from repro.main import build_task_dict  # noqa: E402
from repro.model import WIDTH_PRESETS, Backbone, ShapeDecoder, build_anchor_decoder, build_ppi_decoder  # noqa: E402
from repro.trainer import ResumableTrainer  # noqa: E402
from rri_baselines.protocol import FS, loso_folds, trials_of  # noqa: E402
from rri_baselines.radarode.data import SST_VARIANT, TARGET, CenterBeatWindows  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
# 結果の置き場の接尾辞. 既定(low・center)は 09-28 の結果と同じ場所
VARIANT = ("" if SST_VARIANT == "low" else f"_{SST_VARIANT}") + ("" if TARGET == "center" else f"_{TARGET}")
TRAIN_STRIDE = int(0.8 * FS)          # data_prepared_cft と同じ 0.8 s
TASKS = ["ECG_shape", "PPI", "Anchor"]


def build_trainer(run_dir: Path, epochs: int, loader_gen: torch.Generator):
    params = LibMTL_args.parse_args([])
    params.weighting, params.arch, params.optim, params.scheduler = "EW", "HPS", "sgd", "cos"
    params.lr, params.weight_decay, params.momentum, params.seed = 5e-3, 5e-4, 0.937, 42
    # コサインの周期. 既定はエポック数. 公開実装の main.py は 200 エポックに対して T_max = 100(下がって戻る)
    params.eta_min, params.T_max, params.mode = params.lr * 0.01, int(os.environ.get("RADARODE_TMAX", epochs)), "train"
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
    loader = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=False, num_workers=int(os.environ.get("RADARODE_WORKERS", "4")))
    trainer.model.eval()
    out = []
    for x, _ in loader:
        logits = trainer.model(x.to(trainer.device))["PPI"]
        out.append((logits.squeeze(1).argmax(dim=1) + 1).float().cpu().numpy() / FS * 1000.0)
    return np.concatenate(out)


def main() -> None:
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    if sys.argv[1] == "official":
        # 公開実装 main.py の既定の分割: 試行 1〜88 のうち 75〜85(参加者 29・30)を test, 残りを学習. val はなく test でエポックを選ぶ.
        # 参加者 29・30 の他の試行が学習側に残る(論文の報告値の条件の候補. LOSO との差を測るため)
        test_trials = list(range(75, 86))
        train_trials = [t for t in range(1, 89) if t not in test_trials]
        val_trials, split = test_trials, {"split": "official", "train_trials": train_trials, "test_trials": test_trials}
        run_dir = REPO / "runs" / f"radarode_ew{VARIANT}_official{'' if epochs == 20 else epochs}"
    else:
        index = int(sys.argv[1])
        fold = loso_folds()[index]
        train_trials, val_trials, test_trials = trials_of(fold.train), trials_of(fold.val), trials_of(fold.test)
        split = fold._asdict()
        runs = REPO / "runs" / (f"radarode_ew{VARIANT}" if epochs == 20 else f"radarode_ew{VARIANT}{epochs}")      # 20 エポックの既存の結果と分ける
        run_dir = runs / f"fold{index:02d}_test{fold.test[0]}"
    if (run_dir / "predictions.npz").exists():
        print(f"[skip] {run_dir.name}", flush=True)
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    # 同じ分割を 2 つの作業者が回さないための札(O_EXCL で排他的に作る). 札の PID が死んでいれば取り直して resume.pt から続ける
    claim = run_dir / "claim.pid"
    for _ in range(2):
        try:
            fd = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            break
        except FileExistsError:
            try:
                os.kill(int(claim.read_text() or 0), 0)
                print(f"[skip] {run_dir.name}: 他の作業者が学習中", flush=True)
                return
            except (ProcessLookupError, ValueError, FileNotFoundError):
                claim.unlink(missing_ok=True)
    else:
        return
    if os.environ.get("RADARODE_FAST_OPS", "1") == "1":
        apply_fast_ops()                         # 値を写すだけの処理の置き換え(数値は変わらない, repro/fast_ops.py)
    loader_gen = torch.Generator()
    loader_gen.manual_seed(42)
    train_set = CenterBeatWindows(train_trials, TRAIN_STRIDE)
    val_set = CenterBeatWindows(val_trials, TRAIN_STRIDE)
    mk = lambda ds, sh, g=None: torch.utils.data.DataLoader(  # noqa: E731
        ds, batch_size=32, shuffle=sh, num_workers=int(os.environ.get("RADARODE_WORKERS", "6")),
        pin_memory=int(os.environ.get("RADARODE_WORKERS", "6")) > 0, drop_last=sh, generator=g)  # WORKERS=0: 本体だけで読む(RAM 節約. 2026-10-03)
    trainer = build_trainer(run_dir, epochs, loader_gen)
    n_params = sum(p.numel() for p in trainer.model.parameters())
    (run_dir / "config.json").write_text(json.dumps({"fold": split, "weighting": "EW", "width": "cft", "ode": "on",
                                                     "epochs": epochs, "t_max": int(os.environ.get("RADARODE_TMAX", epochs)), "sst": SST_VARIANT, "target": TARGET, "train_stride_samples": TRAIN_STRIDE,
                                                     "n_train": len(train_set), "n_val": len(val_set),
                                                     "n_params": n_params}, indent=1))
    print(f"[{run_dir.name}] train {len(train_set)} / val {len(val_set)} 窓, {n_params:,} パラメータ, {epochs} エポック", flush=True)
    t0 = time.time()
    # test は毎エポック評価しない(0.2 s 間隔の全窓は学習後に 1 回だけ). 学習器の履歴用には val を渡す
    trainer.train(mk(train_set, True, loader_gen), mk(val_set, False), mk(val_set, False), epochs)
    print(f"[{run_dir.name}] 学習 {time.time() - t0:.0f} s", flush=True)
    trainer.model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location=trainer.device))
    preds = {f"trial{t}_pred_ms": predict(trainer, t) for t in test_trials}
    last = torch.load(run_dir / "resume.pt", map_location=trainer.device, weights_only=False)["model"]
    trainer.model.load_state_dict(last)                  # 最終エポックの模型でも出す(選び方の影響を見る)
    preds.update({f"trial{t}_pred_last_ms": predict(trainer, t) for t in test_trials})
    np.savez(run_dir / "predictions.npz", **preds)
    claim.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
