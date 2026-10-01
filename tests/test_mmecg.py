"""MMECG 再実装の検算: 受容野が原文の 512 であること, キャッシュつき自己回帰が素朴な再計算と一致すること."""
import torch

from rri_baselines.mmecg.model import COND, RECEPTIVE, CachedGenerator, TCNDecoder, class_value, generate, generate_naive


def _decoder(seed: int = 0) -> TCNDecoder:
    torch.manual_seed(seed)
    dec = TCNDecoder(channels=16).double().eval()
    with torch.no_grad():                       # 重みを大きくして番号がばらつくようにする(全部同じ番号だと照合にならない)
        for p in dec.parameters():
            p.mul_(3.0)
    return dec


def test_receptive_field_is_512():
    assert RECEPTIVE == 512


def test_cached_step_matches_teacher_forcing():
    dec = _decoder()
    b, T = 2, 600
    x_prev = torch.rand(b, T, dtype=torch.float64) * 2 - 1
    h = torch.randn(b, COND, T, dtype=torch.float64)
    full = dec(x_prev, h)
    gen = CachedGenerator(dec, b, "cpu", torch.float64)
    stepped = torch.stack([gen.step(x_prev[:, t], h[:, :, t]) for t in range(T)], 2)
    assert (full - stepped).abs().max() < 1e-9


def test_generate_matches_naive():
    dec = _decoder(1)
    h = torch.randn(2, COND, 120, dtype=torch.float64)
    for context in (None, class_value(torch.randint(0, 256, (2, 30))).double()):
        fast = generate(dec, h, context)
        slow, _ = generate_naive(dec, h, context)
        assert torch.equal(fast, slow)
        assert fast.unique().numel() > 1
