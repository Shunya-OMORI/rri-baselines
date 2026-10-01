"""プレーンな系列モデル 3 種(改訂 2 版, 2026-10-02). 入出力の層だけを今回の問題に合わせ, 網の中身は出典のまま.

初版(models_v1_2026-10-01.py)は 50 点を固定チャネルとして畳み込み, 参加者をまたぐ評価で一定値(99.7 ms)と同水準だった. 査読(2026-10-02)で
原因が特定され, 文献調査(同日)で「全手法に同じ前処理・窓・分割・採点を当てる」のが標準と確認されたので, 次の 3 点を直した.
  1. 点ごとの共有重み: MMECG の 50 点の並びは試行ごとに違う(mmwave-rri data_setup/trial.py). 固定チャネルの網は試行固有の並びを覚えるだけになる.
     提案手法と同じく 1 点を 1 チャネルの系列として同じ重みで通し(B·P, 1, 800), 点方向に特徴を平均してから出力の線形層に入れる.
  2. 窓ごと・点ごとの z スコア(研究の標準: Reiss 2019 DeepPPG, Jin 2026 など).
  3. 出力は時間平均(GAP)+ Linear(WaveNet の平坦化 → Linear(800,1) は位置ごとの固定テンプレートで過学習しやすい. TCN は出典どおり最後の時刻).

| 名前 | 出典 | 実装 | 設定(出典の既定) | 入出力の層 |
|---|---|---|---|---|
| tcn | Bai ら 2018, arXiv:1803.01271 | 公式 locuslab/TCN の tcn.py をそのまま(reference/tcn) | 25 ch × 8 段, 核 7, dropout 0.05(公式の pmnist の例), 受容野 3,061 標本 | 最後の時刻の特徴 → 点平均 → Linear(1) |
| wavenet | van den Oord ら 2016, arXiv:1609.03499 | AI-Huang/WaveNet の WaveNetBlock をそのまま(reference/wavenet) | 32 ch, 核 3, 膨張 1〜256 の 9 段, 受容野 1,023 標本 | スキップの和 → ReLU → 1×1 → ReLU → 1×1(同リポジトリの順) → 時間平均 → 点平均 → Linear(1) |
| transformer | Vaswani ら 2017. PyTorch 公式 nn.TransformerEncoder | パッチ埋め込み(Conv1d, 100 ms) + 学習する位置埋め込み + [CLS], Pre-LN | [CLS] の特徴 → 点平均 → Linear(1) |

学習では窓ごとに 50 点から 16 点を無作為に選んで通し(計算量と拡張. 提案手法の点数と同じ), 評価では 50 点すべてを平均する.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_REF = Path(__file__).resolve().parents[3] / "reference"
sys.path.insert(0, str(_REF / "tcn"))
sys.path.insert(0, str(_REF / "wavenet"))
from tcn import TemporalConvNet  # noqa: E402  公式実装をそのまま import
from wavenet import WaveNetBlock  # noqa: E402  AI-Huang/WaveNet をそのまま import

IN_CH = 50


def normalize_points(x: torch.Tensor) -> torch.Tensor:
    """(B, P, 800) → 窓ごと・点ごとの z スコア."""
    return (x - x.mean(dim=2, keepdim=True)) / (x.std(dim=2, keepdim=True) + 1e-6)


class PointwiseRegressor(nn.Module):
    """点ごとに同じ網を通し, 特徴を点で平均して 1 値にする共通の枠. 子クラスは encode((N, 1, 800)) → (N, d) を実装する."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:                     # x: (B, P, 800)
        b, p, n = x.shape
        feat = self.encode(normalize_points(x).reshape(b * p, 1, n))        # (B·P, d)
        return self.head(feat.reshape(b, p, -1).mean(dim=1)).squeeze(1)


class TCNRegressor(PointwiseRegressor):
    """公式 TemporalConvNet(1 入力チャネル) + 最後の時刻の線形読み出し(出典の系列分類 TCN/mnist_pixel/model.py と同じ読み方)."""

    def __init__(self, channels: tuple[int, ...] = (25,) * 8, kernel_size: int = 7, dropout: float = 0.05) -> None:
        super().__init__()
        self.tcn = TemporalConvNet(1, list(channels), kernel_size=kernel_size, dropout=dropout)
        self.head = nn.Linear(channels[-1], 1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.tcn(x)[:, :, -1]


class WaveNetRegressor(PointwiseRegressor):
    """AI-Huang/WaveNet(reference/wavenet)の WaveNetBlock をそのまま使い, 上段は同リポジトリの WaveNet.forward の順
    (スキップの和 → ReLU → 1×1 → ReLU → 1×1)の後, 平坦化 → 全結合の代わりに時間平均 → Linear にする(改訂 2 版)."""

    def __init__(self, filters: int = 32, kernel_size: int = 3, n: int = 9) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(1, filters, 1)
        self.wavenet_block = WaveNetBlock(filters, filters, kernel_size, n)
        self.conv2 = nn.Conv1d(1, 1, 1)
        self.conv3 = nn.Conv1d(1, 1, 1)
        self.head = nn.Linear(1, 1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        _, skips = self.wavenet_block(self.conv1(x))
        y = F.relu(torch.stack(skips, dim=0).sum(dim=0))                     # (N, 1, 800)
        return self.conv3(F.relu(self.conv2(y))).mean(dim=2)                 # (N, 1)


class TransformerRegressor(PointwiseRegressor):
    def __init__(self, d_model: int = 64, n_heads: int = 4, n_layers: int = 3, patch: int = 20, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed = nn.Conv1d(1, d_model, patch, stride=patch)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = nn.Parameter(torch.zeros(1, 800 // patch + 1, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=4 * d_model, dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        tok = self.embed(x).transpose(1, 2)                                  # (N, 40, d)
        tok = torch.cat([self.cls.expand(len(tok), -1, -1), tok], dim=1) + self.pos
        return self.norm(self.encoder(tok))[:, 0]


MODELS = {"tcn": TCNRegressor, "wavenet": WaveNetRegressor, "transformer": TransformerRegressor}


if __name__ == "__main__":
    from torch.utils.flop_counter import FlopCounterMode
    x = torch.zeros(1, IN_CH, 800)
    for name, cls in MODELS.items():
        m = cls().eval()
        with FlopCounterMode(display=False) as fc:
            m(x)
        print(f"{name}: {sum(p.numel() for p in m.parameters()):,} パラメータ, {fc.get_total_flops() / 2 / 1e6:.1f} MMACs/窓(50 点)")
