"""検算: 同じ重み・同じ入力で, 公式モジュールと書き直し版(rri_baselines.mmjepa)の出力が一致するか(CPU).

  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 .venv/bin/python -m rri_baselines.mmjepa_official.check_rewrite
  → results/mmjepa_official/check_rewrite.json

  1. エンコーダ(公式 context_encoder ↔ ContextEncoder): 全トークン / マスクあり(z_pad)
  2. 予測器(公式 attn_MultiMaskPredictor ↔ MultiMaskPredictor): 3 目標と心拍数
  3. マスク(公式 block_mask.mm_mask ↔ masks.mm_mask): 同じ乱数種で同じ配置か
  4. HRC-DiT(公式 HRC_dit ↔ HRCDiT): ε の出力(初期化で 0 になる層があるので全パラメータに小さな乱数を足してから写す)
  5. DDIM 100 段(公式 create_diffusion("100").ddim_sample_loop(eta=0) ↔ Diffusion(respace=100).ddim_sample)
  6. データ: 書き出した npy を公式 npyLoad → reorder_rcg / ecg_patches_to_full / patch_ecg で読んだ入力 ↔
     書き直し版 WindowSet.batch → seg_norm / ecg_to_img(同じ窓)
"""
import json
import random
import tempfile
from pathlib import Path

import numpy as np
import torch

from rri_baselines.mmjepa import masks as rw_masks
from rri_baselines.mmjepa import model as rw
from rri_baselines.mmjepa.data import WindowSet
from rri_baselines.mmjepa.diffusion import Diffusion
from rri_baselines.mmjepa.run import seg_norm
from rri_baselines.mmjepa_official import export as ex
from rri_baselines.mmjepa_official.common import REPO, assert_official_clean, official_on_path

official_on_path()
from block.backbone_multi_target import attn_MultiMaskPredictor, context_encoder  # noqa: E402
from block.block_mask import mm_mask  # noqa: E402
from block.data_load import npyLoad, npyLoadWithHR  # noqa: E402
from custom_model import HRC_dit  # noqa: E402
from diffusion import create_diffusion  # noqa: E402
from end2end_train import ecg_patches_to_full, patch_ecg, reorder_rcg  # noqa: E402

RENAME = [(".sdp_attn.scale", ".scale"), ("W_P.proj.", "W_P."), ("x_embedder.proj.", "x_embedder."),
          (".mlp.fc1.", ".mlp.0."), (".mlp.fc2.", ".mlp.2."), ("final_layer.linear.", "final_linear."),
          ("final_layer.adaLN_modulation.", "final_adaLN."), (".norm_attn.1.", ".norm_attn.bn."),
          (".norm_ffn.1.", ".norm_ffn.bn.")]


def to_rewrite(sd: dict) -> dict:
    out = {}
    for k, v in sd.items():
        for a, b in RENAME:
            k = k.replace(a, b)
        out[k] = v
    return out


def diff(a, b) -> float:
    return float((a.double() - b.double()).abs().max())


def main() -> dict:
    assert_official_clean()
    torch.manual_seed(0)
    res = {}

    # 1. エンコーダ
    off = context_encoder(c_in=50, num_patch=16, patch_len=32, d_model=192, n_layers=2, n_heads=8, d_ff=256, dp=0,
                          device="cpu").eval()
    mine = rw.ContextEncoder(dropout=0.0).eval()
    mine.load_state_dict(to_rewrite(off.state_dict()), strict=True)
    x, pos = torch.randn(3, 16, 50, 32), torch.randn(3, 50, 3)
    with torch.no_grad():
        res["encoder_full"] = diff(off(x, position=pos, obs_m=None), mine(x, pos))
        random.seed(1)
        obs, mt, mc, mb = mm_mask(x, obs=[16, 8], time=2, ch=10, tc=[8, 8])
        s_o, z_o = off(x, position=pos, obs_m=obs)
        s_m, z_m = mine(x, pos, obs)
        res["encoder_masked_s_x"], res["encoder_masked_z_pad"] = diff(s_o, s_m), diff(z_o, z_m)

    # 2. 予測器
    pred_o = attn_MultiMaskPredictor(d_model=192, n_heads=4, d_ff=64, dropout=0.1).eval()
    pred_m = rw.MultiMaskPredictor().eval()
    pred_m.load_state_dict(to_rewrite(pred_o.state_dict()), strict=True)
    with torch.no_grad():
        *h_o, hr_o = pred_o(z_o, (mt, mc, mb))
        h_m, hr_m = pred_m(z_m, (mt, mc, mb))
        res["predictor_targets"] = max(diff(a, b) for a, b in zip(h_o, h_m))
        res["predictor_hr"] = diff(hr_o, hr_m)

    # 3. マスク
    random.seed(7)
    m_o = mm_mask(x, obs=[16, 8], time=2, ch=10, tc=[8, 8])
    m_m = rw_masks.mm_mask(3, 16, 50, rng=random.Random(7))
    res["mask_identical"] = all(torch.equal(a.cpu(), b.cpu()) for a, b in zip(m_o, m_m))

    # 4. HRC-DiT
    dit_o = HRC_dit(learn_sigma=False).eval()
    with torch.no_grad():
        for p in dit_o.parameters():
            if p.requires_grad:
                p.add_(0.02 * torch.randn_like(p))
    dit_m = rw.HRCDiT().eval()
    dit_m.load_state_dict(to_rewrite(dit_o.state_dict()), strict=True)
    xt, t, up = torch.randn(3, 1, 24, 24), torch.tensor([0, 500, 999]), torch.randn(3, 800, 192)
    with torch.no_grad():
        eps_o, eps_m = dit_o(xt, t, up), dit_m(xt, t, up)
    res["dit_eps"] = diff(eps_o, eps_m)
    res["dit_eps_scale"] = float(eps_o.abs().max())

    # 5. DDIM 100 段(η=0, clip)
    z = torch.randn(3, 1, 24, 24)
    with torch.no_grad():
        s_o = create_diffusion(timestep_respacing="100", learn_sigma=False).ddim_sample_loop(
            dit_o.forward, z.shape, z, clip_denoised=True, model_kwargs={"upstream_output": up}, device="cpu", eta=0.0)
        s_m = Diffusion(respace=100).ddim_sample(lambda a, b: dit_m(a, b, up), z)
    res["ddim100_sample"] = diff(s_o, s_m)

    # 6. データの経路
    trial, idx = 1, torch.tensor([0, 5, 40, 100])
    ws = WindowSet([trial], 128, "cpu")
    rcg_m, ecg_m, pos_m, hr_m = ws.batch(idx)
    starts = ex.train_starts(len(ws.rcg))[idx.numpy()]
    from rri_baselines.mmjepa.data import trial_hr_labels
    d = ex.build([(trial, starts, trial_hr_labels(trial, ex.train_starts(len(ws.rcg)))[idx.numpy()])], with_hr=True)
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "d.npy"
        ex.save_npy_dict(f, d)
        ds, ds_hr = npyLoad(str(f)), npyLoadWithHR(str(f))
        rcg_o, ecg_p, pos_o, y_o = (torch.stack(v) for v in zip(*[ds[i] for i in range(len(ds))]))
        hr_o = torch.stack([ds_hr[i][4] for i in range(len(ds_hr))])
    ecg_full = ecg_patches_to_full(ecg_p, patch_len=32)
    res["data_rcg"] = diff(reorder_rcg(rcg_o, 50), seg_norm(rcg_m))
    res["data_rcg_mmJEPA_layout_is_LVP"] = list(rcg_o.shape[1:]) == [16, 50, 32]
    res["data_ecg"] = diff(ecg_full[..., 0], ecg_m)
    res["data_ecg_img"] = diff(patch_ecg(ecg_full), rw.ecg_to_img(ecg_m))
    res["data_pos"] = diff(pos_o, pos_m)
    res["data_hr"] = diff(hr_o, hr_m)
    res["data_y_is_trial"] = bool((y_o == trial).all())

    res["pass"] = all(v < 1e-4 for k, v in res.items() if isinstance(v, float) and k != "dit_eps_scale") and \
        res["mask_identical"] and res["data_rcg_mmJEPA_layout_is_LVP"]
    assert_official_clean()
    out = REPO / "results/mmjepa_official"
    out.mkdir(parents=True, exist_ok=True)
    (out / "check_rewrite.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    main()
