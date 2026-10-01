"""radarODE(Zhang ら, IEEE TMC 24(10), 2025; arXiv:2408.01672)の網: SCEG(Fig. 5, Table I)と長期再構成(Fig. 2(c), Table III, Fig. 7).

SCEG(1 心周期の ECG 200 点を出す)は radarODE-MTL 公開実装の repro/ をそのまま使う(Table I と照合した結果は下の表).
  入力 (B, 50, 71, 120) → backbone(変形畳み込みの ResNet 4 段)→ squeezer (B, 1024, 31) → encoder (B, 128, 863)
  → initial decoder (B, 8, 207) → temporal (B, 1, 200) と ODE(パラメータ推定 → Euler 法 200 歩)→ 掛け算・融合 → (B, 1, 200)
  論文 Table I と公開実装の照合(width="table1" = 段幅 128/256/512/1024 のとき):
    backbone・squeezer・encoder・initial/temporal decoder: 形と核・ストライドが一致(Table I の encoder 3 段目 Cin = 512 は 256 の誤植.
      Table I の入力は 118 フレーム, 公開実装は 120. どちらも squeezer の後は 31 になる)
    ODE decoder: Table I は Linear 4 段(1656→512→128→32→16, 出力 16 = η 15 個 + τ). 公開実装は 8 段(1656→1024→…→15)で τ が無い
    feature fusion: Table I は Conv2d 3 段 + 転置畳み込み 3 段 + 1D 3 段, 公開実装は Conv2d 4 段 + 転置 2D 5 段 + Conv1d 4 段.
      中間の形 (N,32,52)・(N,16,106)・(N,8,10,211) は公開実装の値と一致する(Table I の (N,8,10,211) は 2D のまま残った形)
    → 本文と表で食い違う 2 か所(ODE の段数と τ, 融合の段)は公開実装に合わせる. 表の値どおりに組むと形がつながらない箇所がある
      (融合の (N,64,1,23) から (64,32,5,2) の転置畳み込みでは 52 にならない)ため.

長期再構成(このファイルで新規に書く. Table III・Fig. 6・Fig. 7・p.8 本文):
  入力: 生のレーダ (B, 50, 800) と, SCEG の断片を PPI の長さに伸縮して並べた形態参照 (B, 1, 800)
  encoder: Residual Block(Fig. 6 の変形 2D 畳み込みを 1D 畳み込みに替えたもの)と Downsample Block を 3 組 → (B, 512, 100)
  decoder: 転置畳み込み 3 段 → 時間特徴 (B, 1, 800)
  融合(TCN): 時間特徴と形態参照を 2 チャネルに積む → 膨張 1D 畳み込み 9 段(K = 3, 膨張 2^(l−1))→ (B, 1, 800)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Fig. 7 の 9 段のチャネル列. 論文に値が無いので, 著者の TCN_decoder.py に注記で残っている 9 段の列 [5,5,4,4,3,3,2,2,1] を使う
TCN_CHANNELS = (5, 5, 4, 4, 3, 3, 2, 2, 1)


class ResBlock1d(nn.Module):
    """Table III の Residual Block (Cin, Cout, K, 1). Fig. 6: 畳み込み → BN → ReLU → 入力を足す → 畳み込み(ここでチャネルを変える).
    公開実装の 2D 版(BasicBlock)は 1 つ目が 3×3 の変形畳み込み, 2 つ目が表の K の通常畳み込みなので, 1D でも 1 つ目を K=3, 2 つ目を表の K にする."""

    def __init__(self, cin: int, cout: int, k: int = 5):
        super().__init__()
        self.conv1 = nn.Conv1d(cin, cin, 3, padding=1)
        self.bn = nn.BatchNorm1d(cin)
        self.conv2 = nn.Conv1d(cin, cout, k, padding=k // 2)

    def forward(self, x):
        return self.conv2(F.relu(self.bn(self.conv1(x))) + x)


def down_block(c: int, k: int = 5) -> nn.Module:
    """Table III の Downsample Block (C, C, 5, 2): 畳み込み(ストライド 2)→ BN → ReLU(公開実装の Downsampling の 1D 版)."""
    return nn.Sequential(nn.Conv1d(c, c, k, stride=2, padding=k // 2), nn.BatchNorm1d(c), nn.ReLU())


def up_block(cin: int, cout: int, k: int = 5) -> nn.Module:
    """Table III の Transconv1d Block (Cin, Cout, 5, 2): 転置畳み込みで長さ 2 倍 → BN → ReLU(SCEG の encoder と同じ並び, p.7)."""
    return nn.Sequential(nn.ConvTranspose1d(cin, cout, k, stride=2, padding=k // 2, output_padding=1),
                         nn.BatchNorm1d(cout), nn.ReLU())


class TCNFusion(nn.Module):
    """Fig. 7 の 9 段の膨張畳み込み(K = 3, D = 2 → 膨張 1, 2, …, 256). 2 チャネル → 1 チャネル.
    causal=True で Fig. 7 の絵どおり過去だけを見る. 既定は両側(機械的な振動は R 波より遅れる(p.7)ので, 過去だけでは参照を前へずらせない.
    相手に有利な側). 受容野は 1 + 2·511 = 1023 点(論文 p.8 は 511 と書くが, K = 3 では合わない)."""

    def __init__(self, cin: int = 2, channels=TCN_CHANNELS, k: int = 3, causal: bool = False):
        super().__init__()
        self.causal, self.k = causal, k
        layers, c = [], cin
        for i, co in enumerate(channels):
            layers.append(nn.Conv1d(c, co, k, dilation=2 ** i))
            c = co
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for i, conv in enumerate(self.layers):
            pad = (self.k - 1) * conv.dilation[0]
            x = conv(F.pad(x, (pad, 0) if self.causal else (pad // 2, pad - pad // 2)))
            if i < len(self.layers) - 1:
                x = F.relu(x)
        return x


class TemporalBlock(nn.Module):
    """著者の TCN_decoder.py(radarODE-MTL, "used for MMECG and radarODE")の TemporalBlock の写し: 重み正規化した膨張畳み込み 2 回
    + dropout 0.2 + 残差(チャネルが変わるときは 1×1 畳み込み). causal=False のときは両側を埋める."""

    def __init__(self, cin: int, cout: int, k: int, dilation: int, causal: bool, dropout: float = 0.2):
        super().__init__()
        from torch.nn.utils import weight_norm
        self.pad, self.causal = (k - 1) * dilation, causal
        self.conv1 = weight_norm(nn.Conv1d(cin, cout, k, dilation=dilation))
        self.conv2 = weight_norm(nn.Conv1d(cout, cout, k, dilation=dilation))
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        self.drop = nn.Dropout(dropout)
        self.down = nn.Conv1d(cin, cout, 1) if cin != cout else None
        if self.down is not None:
            self.down.weight.data.normal_(0, 0.01)

    def _p(self, x):
        return F.pad(x, (self.pad, 0) if self.causal else (self.pad // 2, self.pad - self.pad // 2))

    def forward(self, x):
        out = self.drop(F.relu(self.conv1(self._p(x))))
        out = self.drop(F.relu(self.conv2(self._p(out))))
        return F.relu(out + (x if self.down is None else self.down(x)))


class TCNResidual(nn.Module):
    """9 段の TemporalBlock(K = 3, 膨張 2^(l−1), チャネル TCN_CHANNELS). 著者のコードの TCN の組み方."""

    def __init__(self, cin: int = 2, channels=TCN_CHANNELS, k: int = 3, causal: bool = False):
        super().__init__()
        blocks, c = [], cin
        for i, co in enumerate(channels):
            blocks.append(TemporalBlock(c, co, k, 2 ** i, causal))
            c = co
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


class LongTerm(nn.Module):
    """Table III. (rcg (B, 50, 800), 形態参照 (B, 1, 800)) → 4 s の ECG (B, 1, 800).
    tcn="plain"   : 膨張 1D 畳み込みを 9 枚重ねるだけ(Table III・Fig. 7 の文字どおり). 既定
    tcn="residual": 著者の TCN_decoder.py の TemporalBlock を 9 段(残差あり). 最後の 1 チャネルが ReLU で 0 に張り付き,
                    1,500 歩学習しても出力が入力によらず 0 のまま(results/radarode_long/shortcut_fold0_*_residual_1500.json)なので使わない"""

    def __init__(self, in_ch: int = 50, causal_tcn: bool = False, tcn: str = "plain"):
        super().__init__()
        self.encoder = nn.Sequential(
            ResBlock1d(in_ch, 128), down_block(128),            # (B, 128, 400)
            ResBlock1d(128, 256), down_block(256),              # (B, 256, 200)
            ResBlock1d(256, 512), down_block(512))              # (B, 512, 100)
        self.decoder = nn.Sequential(up_block(512, 128), up_block(128, 16), up_block(16, 1))   # (B, 1, 800)
        self.fusion = TCNFusion(2, causal=causal_tcn) if tcn == "plain" else TCNResidual(2, causal=causal_tcn)

    def forward(self, rcg, ref):
        temporal = self.decoder(self.encoder(rcg))
        return self.fusion(torch.cat([temporal, ref], dim=1))


# ------------------------------------------------------------- SCEG -------------------------------------------------------------
def build_sceg(width: str = "table1"):
    """SCEG = radarODE-MTL repro の Backbone + ShapeDecoder(use_ode=True). radarODE-MTL を import できる環境で呼ぶ.
    width="table1": 段幅 128/256/512/1024(radarODE 論文 Table I に書かれた値. 公開実装の既定と同じ)
    width="cft"   : 段幅 32/128/256/512(著者の後続研究 CFT-RFcardi の値. radarODE-MTL の配布重みもこの幅)"""
    import os
    import sys
    from pathlib import Path
    root = Path(os.environ.get("RADARODE_ROOT", "/home/shunya/research/radarODE-MTL"))
    for p in (root, root / "Projects" / "radarODE_plus"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    from repro.model import WIDTH_PRESETS, Backbone, ShapeDecoder
    widths = WIDTH_PRESETS["upstream" if width == "table1" else width]
    return SCEG(Backbone(in_channels=50, widths=widths), ShapeDecoder(dim=widths[-1], use_ode=True))


class SCEG(nn.Module):
    def __init__(self, backbone: nn.Module, shape: nn.Module):
        super().__init__()
        self.backbone, self.shape = backbone, shape

    def forward(self, sst):                     # (B, 50, 71, 120) → (B, 1, 200)
        return self.shape(self.backbone(sst))


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


if __name__ == "__main__":
    m = LongTerm()
    x, r = torch.randn(2, 50, 800), torch.randn(2, 1, 800)
    h = x
    for i, blk in enumerate(m.encoder):
        h = blk(h)
        print(f"encoder {i}: {tuple(h.shape)}")
    for i, blk in enumerate(m.decoder):
        h = blk(h)
        print(f"decoder {i}: {tuple(h.shape)}")
    print(f"出力: {tuple(m(x, r).shape)}  パラメータ: encoder {count_params(m.encoder):,} / decoder {count_params(m.decoder):,} "
          f"/ TCN {count_params(m.fusion):,} / 計 {count_params(m):,}")
