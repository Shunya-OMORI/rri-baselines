"""プレーンな系列モデル 4 種. 入出力だけを今回の問題(4 s 窓の RCG 50 点 (B, 50, 800) → 窓中央の RR[秒] 1 個)に合わせる.

比較の筋(先生の助言 2026-10-01): 他手法の再現実装は原著どおりになる保証がないので, 公開実装のある汎用の網を信頼できる出典から取り,
入出力の層だけを足して, 提案手法と同じ学習条件(窓・損失・最適化・エポック・分割・採点)で回す. 網の中身は出典のまま, 設定は出典の既定値.

| 名前 | 出典 | 実装 | 入出力の層 |
|---|---|---|---|
| tcn | Bai ら 2018, arXiv:1803.01271 | 公式 locuslab/TCN の tcn.py をそのまま(reference/tcn) | 出力は最後の時刻の特徴 → Linear(1)(出典の系列分類と同じ読み方. 因果畳み込みなので最後の時刻が窓全体を見る) |
| wavenet | van den Oord ら 2016, arXiv:1609.03499 §2.1〜2.4 | 公式実装は TensorFlow 内製で非公開. 論文の図 4 のとおりに書く(因果膨張畳み込み・ゲート付き活性・残差・スキップ) | スキップの和 → ReLU → 1×1 → ReLU → 時間平均 → Linear(1)(論文の出力段の softmax を回帰に替える) |
| transformer | Vaswani ら 2017. 実装は PyTorch 公式 nn.TransformerEncoder | パッチ埋め込み(Conv1d, 100 ms = 20 標本) + 学習する位置埋め込み + [CLS] | [CLS] の特徴 → Linear(1) |
| resnet1d | He ら 2016(ResNet-18 の BasicBlock を 1 次元に) | torchvision の BasicBlock の 2D を 1D に置き換え(段数・幅の比は ResNet-18) | 時間平均 → Linear(1) |

受容野はどれも 4 s 窓(800 標本)を覆う(tcn: 核 7 × 6 段で 757 標本, wavenet: 膨張 1〜512 で 1,024 標本).
幅は「パラメータ数が 10^5〜10^6 の範囲」に収まる出典の既定に近い値を 1 つだけ置き, 探索はしない(探索すると比較対象ごとに有利不利が出る).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "reference" / "tcn"))
from tcn import TemporalConvNet  # noqa: E402  公式実装をそのまま import

IN_CH = 50


class TCNRegressor(nn.Module):
    """公式 TemporalConvNet + 最後の時刻の線形読み出し(出典の系列分類 TCN/mnist_pixel/model.py と同じ)."""

    def __init__(self, channels: tuple[int, ...] = (64,) * 6, kernel_size: int = 7, dropout: float = 0.2) -> None:
        super().__init__()
        self.tcn = TemporalConvNet(IN_CH, list(channels), kernel_size=kernel_size, dropout=dropout)
        self.head = nn.Linear(channels[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.tcn(x)[:, :, -1]).squeeze(1)


class WaveNetBlock(nn.Module):
    """図 4 の 1 層: 因果膨張畳み込み → tanh × sigmoid のゲート → 1×1 で残差とスキップ."""

    def __init__(self, res_ch: int, skip_ch: int, dilation: int, kernel_size: int = 2) -> None:
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.filter = nn.Conv1d(res_ch, res_ch, kernel_size, dilation=dilation)
        self.gate = nn.Conv1d(res_ch, res_ch, kernel_size, dilation=dilation)
        self.res = nn.Conv1d(res_ch, res_ch, 1)
        self.skip = nn.Conv1d(res_ch, skip_ch, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        xp = F.pad(x, (self.pad, 0))
        z = torch.tanh(self.filter(xp)) * torch.sigmoid(self.gate(xp))
        return x + self.res(z), self.skip(z)


class WaveNetRegressor(nn.Module):
    def __init__(self, res_ch: int = 32, skip_ch: int = 64, dilations: tuple[int, ...] = tuple(2 ** i for i in range(10))) -> None:
        super().__init__()
        self.inp = nn.Conv1d(IN_CH, res_ch, 1)
        self.blocks = nn.ModuleList(WaveNetBlock(res_ch, skip_ch, d) for d in dilations)
        self.out1 = nn.Conv1d(skip_ch, skip_ch, 1)
        self.out2 = nn.Conv1d(skip_ch, skip_ch, 1)
        self.head = nn.Linear(skip_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, skips = self.inp(x), 0
        for b in self.blocks:
            h, s = b(h)
            skips = skips + s
        y = F.relu(self.out2(F.relu(self.out1(F.relu(skips)))))
        return self.head(y.mean(dim=2)).squeeze(1)


class TransformerRegressor(nn.Module):
    def __init__(self, d_model: int = 64, n_heads: int = 4, n_layers: int = 3, patch: int = 20, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed = nn.Conv1d(IN_CH, d_model, patch, stride=patch)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = nn.Parameter(torch.zeros(1, 800 // patch + 1, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=4 * d_model, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tok = self.embed(x).transpose(1, 2)                                   # (B, 40, d)
        tok = torch.cat([self.cls.expand(len(tok), -1, -1), tok], dim=1) + self.pos
        return self.head(self.norm(self.encoder(tok))[:, 0]).squeeze(1)


class BasicBlock1d(nn.Module):
    """torchvision.models.resnet.BasicBlock の Conv2d/BatchNorm2d を 1 次元にしたもの(核 3, 段の最初で stride 2)."""

    def __init__(self, cin: int, cout: int, stride: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(cin, cout, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(cout)
        self.conv2 = nn.Conv1d(cout, cout, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(cout)
        self.down = None if stride == 1 and cin == cout else nn.Sequential(nn.Conv1d(cin, cout, 1, stride=stride, bias=False), nn.BatchNorm1d(cout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        idt = x if self.down is None else self.down(x)
        return F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x))))) + idt)


class ResNet1dRegressor(nn.Module):
    """ResNet-18 の構成(stem + [2, 2, 2, 2] ブロック)を 1 次元に. 幅は 64/128/256/512 の 1/4(パラメータ数 ~0.7M)."""

    def __init__(self, widths: tuple[int, ...] = (16, 32, 64, 128)) -> None:
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(IN_CH, widths[0], 7, stride=2, padding=3, bias=False), nn.BatchNorm1d(widths[0]), nn.ReLU(),
                                  nn.MaxPool1d(3, stride=2, padding=1))
        layers, cin = [], widths[0]
        for i, w in enumerate(widths):
            layers += [BasicBlock1d(cin, w, 1 if i == 0 else 2), BasicBlock1d(w, w, 1)]
            cin = w
        self.layers = nn.Sequential(*layers)
        self.head = nn.Linear(widths[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.layers(self.stem(x)).mean(dim=2)).squeeze(1)


MODELS = {"tcn": TCNRegressor, "wavenet": WaveNetRegressor, "transformer": TransformerRegressor, "resnet1d": ResNet1dRegressor}


if __name__ == "__main__":
    from torch.utils.flop_counter import FlopCounterMode
    x = torch.zeros(1, IN_CH, 800)
    for name, cls in MODELS.items():
        m = cls().eval()
        with FlopCounterMode(display=False) as fc:
            m(x)
        print(f"{name}: {sum(p.numel() for p in m.parameters()):,} パラメータ, {fc.get_total_flops() / 2 / 1e6:.1f} MMACs/窓")
