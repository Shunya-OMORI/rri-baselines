"""公式の 1 段(事前学習 / DiT 学習 / 推論)が使う活性化メモリを, CPU の最大 RSS の増分から見積もる(GPU を使わずに).

  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 .venv/bin/python -m rri_baselines.mmjepa_official.mem_probe <pretrain|enc|dit|infer> <B>

公式の計算(mmJEPA.py のループ本体, end2end_train.py のループ本体, inference_from_train.py の 1 バッチ)を公式モジュールで
そのままなぞる. 注意スコア (B, 8 頭, 800, 800) を materialize する PatchTST の注意が支配的で, 増分は B にほぼ比例する.
"""
import random
import resource
import sys

import torch
import torch.nn as nn

from rri_baselines.mmjepa_official.common import official_on_path

official_on_path()
from block.backbone_multi_target import attn_MultiMaskPredictor, context_encoder  # noqa: E402
from block.block_mask import mm_mask  # noqa: E402


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def main(kind: str, B: int) -> None:
    torch.manual_seed(0)
    random.seed(0)
    enc = context_encoder(c_in=50, num_patch=16, patch_len=32, d_model=192, n_layers=2, n_heads=8, d_ff=256,
                          dp=0.2 if kind == "pretrain" else 0, device="cpu")
    rcg, pos = torch.randn(B, 16, 50, 32), torch.randn(B, 50, 3)
    if kind == "pretrain":
        teacher = context_encoder(c_in=50, num_patch=16, patch_len=32, d_model=192, n_layers=2, n_heads=8,
                                  d_ff=256, dp=0.2, device="cpu").eval().requires_grad_(False)
        pred = attn_MultiMaskPredictor(d_model=192, n_heads=4, d_ff=64, dropout=0.1)
        base = rss_mb()
        x_obs, mt, mc, mb = mm_mask(rcg, obs=[16, 8], time=2, ch=10, tc=[8, 8])
        _, x_pad = enc(rcg, position=pos, obs_m=x_obs)
        h1, h2, h3, hr = pred(x_pad, (mt, mc, mb))
        with torch.no_grad():
            full = teacher(rcg, position=pos, obs_m=None)
        D = 192
        loss = sum(nn.L1Loss()(h, full[m].view(B, -1, D)) for h, m in zip((h1, h2, h3), (mt, mc, mb))) + hr.pow(2).mean()
        loss.backward()
    elif kind == "enc":            # end2end_train.py の凍結エンコーダ(no_grad なし, パラメータは requires_grad=False)
        enc.eval().requires_grad_(False)
        base = rss_mb()
        up = enc(rcg, position=pos, obs_m=None)
    elif kind in ("dit", "infer"):
        from custom_model import HRC_dit
        from diffusion import create_diffusion
        dit = HRC_dit(learn_sigma=False)
        enc.eval().requires_grad_(False)
        if kind == "infer":
            dit.eval()
        base = rss_mb()
        ctx = torch.no_grad() if kind == "infer" else torch.enable_grad()
        with ctx:
            up = enc(rcg, position=pos, obs_m=None)
            up = up.reshape(B, -1, up.size(-1))
            ecg = torch.randn(B, 1, 24, 24)
            if kind == "dit":
                diff = create_diffusion(timestep_respacing="", learn_sigma=False)
                t = torch.randint(0, 1000, (B,))
                noise = torch.randn_like(ecg)
                loss = diff.training_losses(dit, ecg, t, model_kwargs={"upstream_output": up}, noise=noise)["loss"].mean()
                out = diff.p_mean_variance(dit, diff.q_sample(ecg, t, noise=noise), t, model_kwargs={"upstream_output": up})
                (loss + out["pred_xstart"].abs().mean()).backward()
            else:
                dit(ecg, torch.full((B,), 999), up)
    print(f"{kind} B={B}: 増分 {rss_mb() - base:.0f} MB (1 標本あたり {(rss_mb() - base) / B:.1f} MB)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
