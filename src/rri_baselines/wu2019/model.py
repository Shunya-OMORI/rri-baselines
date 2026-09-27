"""Wu, Sakamoto ら (IEEE Access 2019, 10.1109/ACCESS.2019.2954294) の CNN と入出力.

網(図 1, §III-B): 入力 54×50 → Conv 3×3×5 → Conv 3×3×5 → ReLU → 全結合 2 → 回帰層. プーリングなし.
出力の 2 値は時刻 t の三角波 (W1(t), W2(t)). W1 は R 波で最大・R–R の中点で最小, W2 は S 波で最小・中点で最大(図 4).
入力 φ(t) は t を中心とする変位 27,000 点(6.48 s)を 10 点ごとに平均した 2,700 点を 54×50 に並べたもの(§III-C).
"""
import torch
from torch import nn

ROWS, COLS = 54, 50                      # 図 1 の入力の形
PHI_LEN = ROWS * COLS                    # 2,700 点
FS_IN = 200 * 25 / 12                    # 416.67 Hz. 2,700 点 = 6.48 s(論文の 27,000 点 × 0.24 ms を 10 点平均)


class WuCNN(nn.Module):
    """(b, 2700) → (b, 2). padding なし・stride 1(MATLAB の convolution2dLayer の既定). 23,282 パラメータ."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(1, 5, 3), nn.Conv2d(5, 5, 3), nn.ReLU())
        self.fc = nn.Linear(5 * (ROWS - 4) * (COLS - 4), 2)

    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        # MATLAB の reshape(x, 54, 50) は列優先(1 列 = 連続 54 点)なので, (50, 54) に並べて転置する
        x = phi.reshape(-1, COLS, ROWS).transpose(1, 2).unsqueeze(1)
        return self.fc(self.conv(x).flatten(1))
