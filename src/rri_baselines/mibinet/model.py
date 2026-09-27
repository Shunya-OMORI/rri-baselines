"""MIBINET (Murshed ら, Array 2023, arXiv 2211.07776) の網と損失.

網(Fig. 7, p.7): BN → Conv1D 64/50 ×2 → MaxPool 5 → BN → Conv1D 32/25 ×2 → SepConv1D 64/32・32/32 → MaxPool 7
→ Flatten → Dense 200 → BN → Dense 100 → Dense 50 → Dense 出力. 畳み込みは Swish, 全結合は ReLU. stride 1・valid
(この形で論文の 1.09M と一致する. BN は Flatten の直後でなく Dense 200 の後に置くと合う).
MMECG への移し方(specs/mibinet2023.md §7.3 の本線): 入力は 4 s 窓の RCG 50 点を 500 Hz に補間した (50, 2000), 出力は中心の拍の RR 1 個.
"""
import torch
from torch import nn


class SeparableConv1d(nn.Sequential):
    """Keras の SeparableConv1D(depth_multiplier 1): 深さ方向の畳み込み(バイアスなし) → 1×1 の畳み込み."""

    def __init__(self, cin: int, cout: int, k: int) -> None:
        super().__init__(nn.Conv1d(cin, cin, k, groups=cin, bias=False), nn.Conv1d(cin, cout, 1))


class MIBINet(nn.Module):
    def __init__(self, in_channels: int = 50, length: int = 2000, n_out: int = 1) -> None:
        super().__init__()
        act = nn.SiLU()                                  # Swish
        self.features = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.Conv1d(in_channels, 64, 50), act, nn.Conv1d(64, 64, 50), act, nn.MaxPool1d(5),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 32, 25), act, nn.Conv1d(32, 32, 25), act,
            SeparableConv1d(32, 64, 32), act, SeparableConv1d(64, 32, 32), act, nn.MaxPool1d(7),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flat = self.features(torch.zeros(2, in_channels, length)).shape[1]
        self.head = nn.Sequential(
            nn.Linear(n_flat, 200), nn.ReLU(), nn.BatchNorm1d(200),
            nn.Linear(200, 100), nn.ReLU(), nn.Linear(100, 50), nn.ReLU(), nn.Linear(50, n_out), nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


# 損失の重み(p.8 §4.2)と非対称 Huber の係数(式 (3), p.7)
W_CORR, W_HUBER, W_MSE, W_MAE, W_AHUBER = 0.002, 1.0, 0.0096, 0.002, 0.0032
PSI, RHO = 1.0, 1.0


def mibinet_loss(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    """式 (2): w1 (1 − r²) + w2 Huber + w3 ε² + w4 |ε| + w5 非対称 Huber. 単位は秒, バッチで平均."""
    e = (pred - true).flatten()
    pc, tc = pred.flatten() - pred.mean(), true.flatten() - true.mean()
    r = (pc * tc).sum() / (pc.norm() * tc.norm() + 1e-8)
    huber = nn.functional.huber_loss(pred.flatten(), true.flatten(), delta=1.0)
    a = e.abs()
    ahuber = torch.where(a <= PSI, 0.5 * (e ** 2 + a + RHO * e), 2 * PSI * a + RHO * PSI * e - 0.5 * PSI ** 2).mean()
    return W_CORR * (1 - r ** 2) + W_HUBER * huber + W_MSE * (e ** 2).mean() + W_MAE * a.mean() + W_AHUBER * ahuber
