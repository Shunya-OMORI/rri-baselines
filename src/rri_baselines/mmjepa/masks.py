"""多粒度マスク(公式 block/block_mask.py の mm_mask @ 4423ef6 の書き直し).

公式 mmJEPA.py の呼び方 mm_mask(rcg, obs=[16, 8], time=2, ch=10, tc=[8, 8]) を既定にする. 論文 p.38966
「Multi-granularity masking with (2 time, 10 channel, 8 × 8 block)」. 形は (B, L=16 時間パッチ, V=50 点).

- 観測 obs: 高さ 16(時間すべて)× 幅 8 の連続した点. 文脈エンコーダはここだけを見る
- 「time」目標: 観測外の連続 2 点 × 時間すべて(公式の名前は time だが, 列 = 点を丸ごと隠す)
- 「channel」目標: 時間 1 行 × 観測外・time 目標外の 10 点(ばらばら)
- 「block」目標: 観測と重ならない 8 × 8 の塊(time/channel 目標とは重なってよい)
"""
import random

import torch


def mm_mask(B: int, L: int, V: int, obs=(16, 8), time=2, ch=10, tc=(8, 8), max_retry=20, device="cpu",
            rng: random.Random | None = None):
    rng = rng or random
    OBS_H, OBS_W = obs
    T3_H, T3_W = tc
    obs_m = torch.zeros(B, L, V, dtype=torch.bool)
    t1_m, t2_m, t3_m = torch.zeros_like(obs_m), torch.zeros_like(obs_m), torch.zeros_like(obs_m)
    for b in range(B):
        for _ in range(max_retry):
            top, left = rng.randint(0, L - OBS_H), rng.randint(0, V - OBS_W)
            free = sorted(set(range(V)) - set(range(left, left + OBS_W)))
            if len(free) < time + ch:
                continue
            valid = [c for c in free if c + time <= V]
            if not valid:
                continue
            c1 = rng.choice(valid)
            free2 = sorted(set(free) - set(range(c1, c1 + time)))
            if len(free2) < ch:
                continue
            row = rng.randint(0, L - 1)
            cols = rng.sample(free2, ch)
            obs_m[b, top:top + OBS_H, left:left + OBS_W] = True
            t1_m[b, :, c1:c1 + time] = True
            t2_m[b, row, cols] = True
            break
        else:
            raise RuntimeError("mm_mask: 観測・time・channel の配置に失敗")
        for _ in range(max_retry):
            r0, c0 = rng.randint(0, L - T3_H), rng.randint(0, V - T3_W)
            if not obs_m[b, r0:r0 + T3_H, c0:c0 + T3_W].any():
                t3_m[b, r0:r0 + T3_H, c0:c0 + T3_W] = True
                break
        else:
            raise RuntimeError("mm_mask: block の配置に失敗")
    return tuple(m.to(device) for m in (obs_m, t1_m, t2_m, t3_m))
