"""MMECG (Chen ら, IEEE TMC 23(1):270-285, 2024) の ECG 再構成モデル. 構成は p.277 Fig. 6 と p.279 §4.2.3.

エンコーダ(1 回で 640 標本 = 3.2 s の RCG 50 点から条件 h (4, 640) を作る)
  時間方向 CNN : 点ごとに (1, 640) → 2 層の畳み込み(核 7, 同じ長さ, 各層 ReLU → BN)を 4 回. 最初の 3 回の後に最大値プーリング 1/2
                 → (32, 80). Fig. 6 の (N, 32, T/8) に合わせてプーリングは 3 回(4 回だと T/16 = 40 になる)
  空間方向     : (32, 80) を平らにして線形で 32 次元へ, 位置 posXYZ (3) も線形で 32 次元へ埋めて足す → Transformer 3 ブロック
                 (4 頭, Q/K/V 64 次元, FF 128) → (50, 32)
  展開         : 時間方向は 2 本目の CNN 4 層(転置畳み込み → 核 7 の畳み込み 2 回. 特徴の数を半分に)で (32, 80) → (4, 640),
                 空間方向は線形で 32 → 4×640. 両者の要素ごとの積(dot product)を取り, (50·4) → 4 の線形で (4, 640) にする
デコーダ(式 (13) p(x_t | x_<t, h_t) を 1 標本ずつ)
  TCN 9 段(核 2, 膨張 1, 2, …, 256 → 受容野 1 + Σ d = 512. p.279 の「9 stacks, dilation 2, 受容野 512」と一致させるため
  1 段 1 畳み込み). 入力は 1 つ前の標本の μ 則の値(1 チャネル)と h_t (4 チャネル). 出力は μ 則 256 値の分類(式 (14))
学習は teacher forcing(1 回の順伝播で全位置を並列に出す). 推論は 1 標本ずつの自己回帰で, 各段の過去の入力を
リングバッファに持つ(fast WaveNet 方式. 素朴な全系列の再計算と数値が一致することを tests/test_mmecg.py で確かめる).
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

N_POINTS, SEG, N_CLASSES, MU = 50, 640, 256, 255
FEAT, COND = 32, 4                    # 時間方向 CNN の出力チャネル, 条件 h のチャネル
TCN_LAYERS, TCN_KERNEL = 9, 2
RECEPTIVE = 1 + (TCN_KERNEL - 1) * (2 ** TCN_LAYERS - 1)   # 512


# ---- μ 則(式 (14), G.711)---------------------------------------------------------------------------
def mu_law_classes(x: torch.Tensor) -> torch.Tensor:
    """[-1, 1] の ECG → 256 値の番号 (0..255)."""
    y = torch.sign(x) * torch.log1p(MU * x.abs()) / math.log1p(MU)
    return ((y + 1) / 2 * MU + 0.5).floor().clamp(0, MU).long()


def class_value(c: torch.Tensor) -> torch.Tensor:
    """番号 → 圧伸した値 [-1, 1](TCN への入力)."""
    return c.float() * (2.0 / MU) - 1.0


def mu_law_decode(c: torch.Tensor) -> torch.Tensor:
    """番号 → ECG [-1, 1](圧伸を戻す)."""
    y = class_value(c)
    return torch.sign(y) * torch.expm1(y.abs() * math.log1p(MU)) / MU


# ---- エンコーダ ---------------------------------------------------------------------------------------
def conv_relu_bn(cin: int, cout: int) -> list[nn.Module]:
    """§4.2.3「Each convolution is followed by a ReLU and batch normalization」の順."""
    return [nn.Conv1d(cin, cout, 7, padding=3), nn.ReLU(), nn.BatchNorm1d(cout)]


class TemporalEncoder(nn.Sequential):
    """(b·50, 1, 640) → (b·50, 32, 80)."""

    def __init__(self) -> None:
        layers = []
        for i in range(4):
            layers += conv_relu_bn(1 if i == 0 else FEAT, FEAT) + conv_relu_bn(FEAT, FEAT)
            if i < 3:
                layers.append(nn.MaxPool1d(2, 2))
        super().__init__(*layers)


class TemporalExpander(nn.Sequential):
    """(b·50, 32, 80) → (b·50, 4, 640). 各層は転置畳み込みで始まり, 核 7 の畳み込み 2 回で特徴を半分にする.
    80 → 640 は 3 回の 2 倍, 32 → 4 は 3 回の半減なので, 4 層目は長さ 640・4 チャネルのまま(転置畳み込みは核 1)."""

    def __init__(self) -> None:
        layers = []
        for cin, cout, up in ((32, 16, 2), (16, 8, 2), (8, 4, 2), (4, 4, 1)):
            layers.append(nn.ConvTranspose1d(cin, cin, up, stride=up))
            layers += conv_relu_bn(cin, cout) + conv_relu_bn(cout, cout)
        super().__init__(*layers)


class TransformerBlock(nn.Module):
    """Vaswani らの後置 LN のブロック. d_model 32, 4 頭, Q/K/V はそれぞれ 64 次元(1 頭 16), FF 128."""

    def __init__(self, d: int = 32, heads: int = 4, qkv: int = 64, ff: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.heads, self.qkv = heads, qkv
        self.q, self.k, self.v, self.o = nn.Linear(d, qkv), nn.Linear(d, qkv), nn.Linear(d, qkv), nn.Linear(qkv, d)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.ReLU(), nn.Dropout(dropout), nn.Linear(ff, d))
        self.ln1, self.ln2, self.drop = nn.LayerNorm(d), nn.LayerNorm(d), nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:          # (b, 50, 32)
        b, n, _ = x.shape
        split = lambda t: t.view(b, n, self.heads, self.qkv // self.heads).transpose(1, 2)
        a = F.scaled_dot_product_attention(split(self.q(x)), split(self.k(x)), split(self.v(x)),
                                           dropout_p=self.drop.p if self.training else 0.0)
        x = self.ln1(x + self.drop(self.o(a.transpose(1, 2).reshape(b, n, self.qkv))))
        return self.ln2(x + self.drop(self.ff(x)))


class Encoder(nn.Module):
    """RCG (b, 50, 640) と位置 (b, 50, 3) → 条件 h (b, 4, 640)."""

    def __init__(self) -> None:
        super().__init__()
        self.temporal = TemporalEncoder()
        self.proj = nn.Linear(FEAT * SEG // 8, FEAT)                  # (32, 80) を平らにして 32 へ
        self.pos = nn.Linear(3, FEAT)                                 # 50×3 → 50×32
        self.transformer = nn.Sequential(*[TransformerBlock() for _ in range(3)])
        self.expand_t = TemporalExpander()
        self.expand_s = nn.Linear(FEAT, COND * SEG)                    # 32 → 4×640
        self.fuse = nn.Linear(N_POINTS * COND, COND)                  # (50·4) → 4 を時刻ごとに

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        b, n, t = x.shape
        ft = self.temporal(x.reshape(b * n, 1, t))                                   # (b·50, 32, 80)
        fs = self.transformer(self.proj(ft.reshape(b, n, -1)) + self.pos(pos))       # (b, 50, 32)
        et = self.expand_t(ft).reshape(b, n, COND, t)                                # (b, 50, 4, 640)
        es = self.expand_s(fs).reshape(b, n, COND, t)                                # (b, 50, 4, 640)
        return self.fuse((et * es).permute(0, 3, 1, 2).reshape(b, t, n * COND)).transpose(1, 2)


# ---- デコーダ(TCN)------------------------------------------------------------------------------------
class CausalBlock(nn.Module):
    """因果的な膨張畳み込み(核 2)1 本と残差(Bai らの TCN のブロックを 1 畳み込みにしたもの)."""

    def __init__(self, cin: int, cout: int, dilation: int) -> None:
        super().__init__()
        self.d = dilation
        self.conv = nn.Conv1d(cin, cout, TCN_KERNEL, dilation=dilation)
        self.res = nn.Conv1d(cin, cout, 1) if cin != cout else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:          # (b, cin, T) → (b, cout, T)
        y = F.relu(self.conv(F.pad(x, (self.d, 0))))
        return F.relu(y + (x if self.res is None else self.res(x)))

    def step(self, x: torch.Tensor, past: torch.Tensor) -> torch.Tensor:
        """x (b, cin) は時刻 t の入力, past (b, cin) は t - d の入力. 出力 (b, cout)."""
        w = self.conv.weight                                      # (cout, cin, 2): [:, :, 0] が t - d, [:, :, 1] が t
        y = F.relu(past @ w[:, :, 0].T + x @ w[:, :, 1].T + self.conv.bias)
        return F.relu(y + (x if self.res is None else x @ self.res.weight[:, :, 0].T + self.res.bias))


class TCNDecoder(nn.Module):
    def __init__(self, channels: int = 128) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([CausalBlock(1 + COND if i == 0 else channels, channels, 2 ** i)
                                     for i in range(TCN_LAYERS)])
        self.head = nn.Conv1d(channels, N_CLASSES, 1)

    def forward(self, x_prev: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """teacher forcing. x_prev (b, T) は 1 つ前の標本の圧伸値, h (b, 4, T) → logits (b, 256, T)."""
        z = torch.cat([x_prev[:, None], h], 1)
        for blk in self.blocks:
            z = blk(z)
        return self.head(z)


class MMECG(nn.Module):
    def __init__(self, tcn_channels: int = 128) -> None:
        super().__init__()
        self.encoder = Encoder()
        self.decoder = TCNDecoder(tcn_channels)

    def forward(self, rcg: torch.Tensor, pos: torch.Tensor, x_prev: torch.Tensor) -> torch.Tensor:
        return self.decoder(x_prev, self.encoder(rcg, pos))


# ---- 自己回帰の生成 -----------------------------------------------------------------------------------
class CachedGenerator:
    """TCN を 1 標本ずつ進める. 段 i の入力の過去 d_i 個をリングバッファ (b, cin, d_i) に持つ.
    バッファの初期値 0 は学習時の因果的なゼロ埋め(F.pad)と同じ."""

    def __init__(self, dec: TCNDecoder, batch: int, device, dtype=torch.float32) -> None:
        self.dec, self.t = dec, 0
        self.bufs = [torch.zeros(batch, blk.conv.in_channels, blk.d, device=device, dtype=dtype) for blk in dec.blocks]

    def step(self, x_prev: torch.Tensor, h_t: torch.Tensor) -> torch.Tensor:
        """x_prev (b,) 圧伸値, h_t (b, 4) → logits (b, 256)."""
        z = torch.cat([x_prev[:, None], h_t], 1)
        for blk, buf in zip(self.dec.blocks, self.bufs):
            k = self.t % blk.d
            past = buf[:, :, k].clone()
            buf[:, :, k] = z
            z = blk.step(z, past)
        self.t += 1
        return z @ self.dec.head.weight[:, :, 0].T + self.dec.head.bias


@torch.no_grad()
def generate(dec: TCNDecoder, h: torch.Tensor, context: torch.Tensor | None = None, sample: bool = False,
             generator: torch.Generator | None = None) -> torch.Tensor:
    """h (b, 4, T) を条件に T 標本を 1 標本ずつ生成し, 番号 (b, T) を返す.
    context (b, L) は生成の前に与える過去の圧伸値(その区間の h は 0). None なら学習時と同じゼロ埋め."""
    b, _, T = h.shape
    gen = CachedGenerator(dec, b, h.device, h.dtype)
    x_prev = torch.zeros(b, device=h.device, dtype=h.dtype)
    if context is not None:
        zero_h = torch.zeros(b, COND, device=h.device, dtype=h.dtype)
        for j in range(context.shape[1]):
            gen.step(x_prev, zero_h)
            x_prev = context[:, j]
    out = torch.empty(b, T, dtype=torch.long, device=h.device)
    for t in range(T):
        logits = gen.step(x_prev, h[:, :, t])
        c = torch.multinomial(F.softmax(logits, -1), 1, generator=generator)[:, 0] if sample else logits.argmax(-1)
        out[:, t] = c
        x_prev = class_value(c).to(h.dtype)
    return out


@torch.no_grad()
def generate_naive(dec: TCNDecoder, h: torch.Tensor, context: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """照合用の素朴な自己回帰: 各時刻で, それまでの入力系列全体に TCN の順伝播(teacher forcing と同じ関数)をかけ直す.
    番号 (b, T) と各時刻の logits (b, 256, T) を返す. 貪欲(argmax)のみ."""
    b, _, T = h.shape
    L = 0 if context is None else context.shape[1]
    xs = torch.zeros(b, L + T, device=h.device, dtype=h.dtype)   # 位置 j の入力 = 1 つ前の標本の値(先頭は 0)
    if L:
        xs[:, 1:L + 1] = context
    hs = torch.cat([torch.zeros(b, COND, L, device=h.device, dtype=h.dtype), h], 2)
    out = torch.empty(b, T, dtype=torch.long, device=h.device)
    logits_all = torch.empty(b, N_CLASSES, T, device=h.device, dtype=h.dtype)
    for t in range(T):
        logits = dec(xs[:, :L + t + 1], hs[:, :, :L + t + 1])[:, :, -1]
        logits_all[:, :, t] = logits
        out[:, t] = logits.argmax(-1)
        if L + t + 1 < L + T:
            xs[:, L + t + 1] = class_value(out[:, t]).to(h.dtype)
    return out, logits_all
