"""mm-JEPA-ECG(Liu ら, AAAI-26)のネットワーク. 公式実装 lanyangyang/mmJEPA-ECG @ 4423ef6 の書き直し.

公式リポジトリにはライセンスの記載が無い(既定で全権利留保)ため, コードは写さず, 同じ数式・同じ既定値で書き直した.
対応する公式ファイルは各クラスの docstring に書く. timm の Attention / Mlp / PatchEmbed は同じ計算の最小版で置き換えた
(パラメータの形・数は同じ).

入出力の次元(MMECG, 論文 p.38966「512-sample windows with 50 channels … patch size and stride of 32」):
  レーダ窓 (B, 512, 50) → パッチ (B, L=16, V=50, P=32) → エンコーダ → (B, 16, 50, 192)
  ECG 512 標本 → 576 に 0 詰め → (B, 1, 24, 24) の「画像」 → DiT(4×4 パッチ, 36 トークン) → ε (B, 1, 24, 24)
"""
import math
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

N_PATCH, PATCH_LEN, N_VARS = 16, 32, 50
ENC_DIM, ENC_FF, ENC_HEADS, ENC_LAYERS = 192, 256, 8, 2     # 論文 p.38966
ECG_LEN, ECG_IMG = 512, 24                                   # 512 → 24×24=576(公式 patch_ecg)


# ---------------------------------------------------------------- エンコーダ(block/backbone_multi_target.py)
def build_2d_sincos_pe(L: int, V: int, D: int) -> torch.Tensor:
    """時間 L × チャネル V の固定位置符号 (L, V, D). 前半 D/2 が時間, 後半 D/2 がチャネル(block/Positionemb.py)."""
    dim = D // 2
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2) / dim))
    pe_t = torch.arange(L)[:, None] * inv_freq
    pe_c = torch.arange(V)[:, None] * inv_freq
    pe = torch.zeros(L, V, D)
    pe[..., :dim] = torch.cat([pe_t.sin(), pe_t.cos()], -1)[:, None, :]
    pe[..., dim:] = torch.cat([pe_c.sin(), pe_c.cos()], -1)[None, :, :]
    return pe


class TSTAttention(nn.Module):
    """PatchTST の多頭注意(block/attention.py). res_attention=True のとき前の層の注意スコアを足す."""

    def __init__(self, d_model: int, n_heads: int, res_attention: bool, proj_dropout: float):
        super().__init__()
        self.h, self.dk = n_heads, d_model // n_heads
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.res_attention = res_attention
        # 公式は nn.Parameter(requires_grad=False)(lsa=False). パラメータ数に 1 個ずつ入るので同じ形で持つ
        self.scale = nn.Parameter(torch.tensor(self.dk ** -0.5), requires_grad=False)
        self.to_out = nn.Sequential(nn.Linear(d_model, d_model), nn.Dropout(proj_dropout))

    def forward(self, x: torch.Tensor, prev: Optional[torch.Tensor] = None):
        b, n, _ = x.shape
        q = self.W_Q(x).view(b, n, self.h, self.dk).transpose(1, 2)
        k = self.W_K(x).view(b, n, self.h, self.dk).transpose(1, 2)
        v = self.W_V(x).view(b, n, self.h, self.dk).transpose(1, 2)
        if self.res_attention:
            scores = (q @ k.transpose(-1, -2)) * self.scale
            if prev is not None:
                scores = scores + prev
            out = scores.softmax(-1) @ v
        else:           # スコアを次の層へ渡さないときは同じ計算を省メモリの SDPA で行う(scale は同じ値)
            scores = None
            out = F.scaled_dot_product_attention(q, k, v, scale=float(self.scale))
        out = self.to_out(out.transpose(1, 2).reshape(b, n, -1))
        return out, scores


class TSTEncoderLayer(nn.Module):
    """公式 TSTEncoderLayer. pre_norm=True では「正規化した src」に残差を足す(公式の実装どおり)."""

    def __init__(self, d_model, n_heads, d_ff, norm="BatchNorm", dropout=0.0, res_attention=False, pre_norm=False):
        super().__init__()
        self.self_attn = TSTAttention(d_model, n_heads, res_attention, dropout)
        self.dropout_attn = nn.Dropout(dropout)
        self.norm_attn = _norm(norm, d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model))
        self.dropout_ffn = nn.Dropout(dropout)
        self.norm_ffn = _norm(norm, d_model)
        self.pre_norm = pre_norm

    def forward(self, src, prev=None):
        if self.pre_norm:
            src = self.norm_attn(src)
        src2, scores = self.self_attn(src, prev)
        src = src + self.dropout_attn(src2)
        if not self.pre_norm:
            src = self.norm_attn(src)
        if self.pre_norm:
            src = self.norm_ffn(src)
        src = src + self.dropout_ffn(self.ff(src))
        if not self.pre_norm:
            src = self.norm_ffn(src)
        return src, scores


class _BatchNormTokens(nn.Module):
    """公式の Transpose(1,2) → BatchNorm1d → Transpose(1,2)."""

    def __init__(self, d):
        super().__init__()
        self.bn = nn.BatchNorm1d(d)

    def forward(self, x):
        return self.bn(x.transpose(1, 2)).transpose(1, 2)


def _norm(kind: str, d: int) -> nn.Module:
    return _BatchNormTokens(d) if "batch" in kind.lower() else nn.LayerNorm(d)


class TSTEncoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, n_layers, norm="BatchNorm", dropout=0.0, res_attention=False,
                 pre_norm=False):
        super().__init__()
        self.layers = nn.ModuleList([TSTEncoderLayer(d_model, n_heads, d_ff, norm, dropout, res_attention, pre_norm)
                                     for _ in range(n_layers)])

    def forward(self, x):
        scores = None
        for layer in self.layers:
            x, scores = layer(x, scores)
        return x


class ContextEncoder(nn.Module):
    """公式 context_encoder. 2 層の Patch Transformer(d=192, FF 256, 8 頭, LayerNorm, pre-norm, 残差注意).

    x (B, L, V, P) → パッチごとの線形射影 + 2D 位置符号 + 点の 3 次元座標の射影 → (B, L·V, D) を注意 → (B, L, V, D).
    obs_m を渡すと観測トークンだけを通し, 残りは mask_token(勾配なし, 公式どおり .data)で埋めた z_pad も返す.
    """

    def __init__(self, c_in=N_VARS, num_patch=N_PATCH, patch_len=PATCH_LEN, n_layers=ENC_LAYERS, d_model=ENC_DIM,
                 n_heads=ENC_HEADS, d_ff=ENC_FF, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.W_P = nn.Linear(patch_len, d_model, bias=False)
        self.register_buffer("W_pos", build_2d_sincos_pe(num_patch, c_in, d_model))
        self.space_proj = nn.Linear(3, d_model, bias=False)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, d_model))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.encoder = TSTEncoder(d_model, n_heads, d_ff, n_layers, norm="LayerNorm", dropout=dropout,
                                  res_attention=True, pre_norm=True)

    def forward(self, x, position, obs_m=None):
        B, L, V, _ = x.shape
        D = self.d_model
        x = self.W_P(x) + self.W_pos
        sp = self.space_proj(position)[:, None, :, :]
        x = x + sp
        if obs_m is None:
            return self.encoder(x.reshape(B, L * V, D)).view(B, L, V, D)
        s_x = self.encoder(x[obs_m].view(B, -1, D))
        z_pad = (self.mask_token.detach() + self.W_pos).expand(B, L, V, D) + sp
        z_pad = z_pad.masked_scatter(obs_m.unsqueeze(-1).expand(-1, -1, -1, D), s_x.reshape(-1))
        return s_x, z_pad


class MultiMaskPredictor(nn.Module):
    """公式 attn_MultiMaskPredictor. 1 層の Patch Transformer(d=192, FF 64, 4 頭, BatchNorm, pre-norm, dropout 0.1)
    と心拍数ヘッド(全トークン平均 → 192→96→48→1). 論文 p.38967「single-layer Patch Transformer (d=192, dff=64, 4 heads)
    with an MLP head」. query_tok は公式にあって使われない(パラメータ数を合わせるため残す)."""

    def __init__(self, d_model=ENC_DIM, n_heads=ENC_HEADS // 2, d_ff=ENC_FF // 4, dropout=0.1):
        super().__init__()
        self.query_tok = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.query_tok, std=0.02)
        self.head = TSTEncoder(d_model, n_heads, d_ff, 1, norm="BatchNorm", dropout=dropout, pre_norm=True)
        self.hr_predictor = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Dropout(dropout),
                                          nn.Linear(d_model // 2, d_model // 4), nn.GELU(), nn.Linear(d_model // 4, 1))

    def forward(self, z_pad, masks):
        B, L, V, D = z_pad.shape
        out = self.head(z_pad.reshape(B, L * V, D)).view(B, L, V, D)
        hr = self.hr_predictor(out.mean(dim=(1, 2)))
        return [out[m].view(B, -1, D) for m in masks], hr


# ---------------------------------------------------------------- HRC-DiT(custom_model.py)
def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden, freq_dim=256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(nn.Linear(freq_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden))

    def forward(self, t):
        half = self.freq_dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
        args = t[:, None].float() * freqs[None]
        return self.mlp(torch.cat([args.cos(), args.sin()], -1))


class SelfAttention(nn.Module):
    """timm.layers.Attention(qkv_bias=True, qk_norm なし)と同じ計算."""

    def __init__(self, dim, heads):
        super().__init__()
        self.h = heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.h, C // self.h).permute(2, 0, 3, 1, 4)
        return self.proj(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, N, C))


class CrossAttention(nn.Module):
    """公式 CrossAttention(q/k/v はバイアスなし, 6 頭 × 64). 文脈(レーダ 800 トークン)の k, v は拡散の段に依らないので
    kv_cache を渡すと使い回す(同じ計算. 公式は毎段計算し直す. FLOPs は公式の数え方で出す)."""

    def __init__(self, dim, heads):
        super().__init__()
        self.h, self.dh = heads, dim // heads
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(0.0))

    def kv(self, ctx):
        B, M, C = ctx.shape
        return (self.to_k(ctx).view(B, M, self.h, self.dh).transpose(1, 2),
                self.to_v(ctx).view(B, M, self.h, self.dh).transpose(1, 2))

    def forward(self, x, ctx, kv=None):
        B, N, C = x.shape
        k, v = kv if kv is not None else self.kv(ctx)
        q = self.to_q(x).view(B, N, self.h, self.dh).transpose(1, 2)
        attn = ((q @ k.transpose(-1, -2)) * self.dh ** -0.5).softmax(-1)
        return self.to_out((attn @ v).transpose(1, 2).reshape(B, N, C))


class HRCBlock(nn.Module):
    """公式 DiTBlock_CrossAttention_LN. 式 (11): H = MLCA(MHSA(LN(e)) + e, z). adaLN は自己注意と MLP にだけ掛かる."""

    def __init__(self, hidden, heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.attn = SelfAttention(hidden, heads)
        self.norm2 = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.crossAtten = CrossAttention(hidden, heads)
        self.norm3 = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(hidden, mlp_hidden), nn.GELU(approximate="tanh"), nn.Linear(mlp_hidden, hidden))
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 6 * hidden))

    def forward(self, x, up, t, kv=None):
        sh_msa, sc_msa, g_msa, sh_mlp, sc_mlp, g_mlp = self.adaLN_modulation(t).chunk(6, dim=1)
        x = x + g_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), sh_msa, sc_msa))
        x = x + self.crossAtten(self.norm2(x), up, kv)
        x = x + g_mlp.unsqueeze(1) * self.mlp(modulate(self.norm3(x), sh_mlp, sc_mlp))
        return x


class HRCDiT(nn.Module):
    """公式 HRC_dit(): depth 6, hidden 384, 6 頭, MLP 4 倍(=1536), patch 4, 入力 24×24, 文脈 192 次元, learn_sigma=False.
    論文 p.38967「stacked 6 HRC-DiT blocks (dmodel=384, dff=1536, 6 heads)」."""

    def __init__(self, input_size=ECG_IMG, patch_size=4, hidden=384, depth=6, heads=6, mlp_ratio=4.0, upstream_dim=ENC_DIM):
        super().__init__()
        self.p = patch_size
        self.grid = input_size // patch_size
        self.x_embedder = nn.Conv2d(1, hidden, patch_size, stride=patch_size)     # timm PatchEmbed と同じ
        self.upstream_embedder = nn.Linear(upstream_dim, hidden)
        self.t_embedder = TimestepEmbedder(hidden)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.grid ** 2, hidden), requires_grad=False)
        self.blocks = nn.ModuleList([HRCBlock(hidden, heads, mlp_ratio) for _ in range(depth)])
        self.final_norm = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.final_linear = nn.Linear(hidden, patch_size * patch_size)
        self.final_adaLN = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 2 * hidden))
        self._init_weights()

    def _init_weights(self):
        """公式 DiT.initialize_weights と同じ: Linear は xavier_uniform・バイアス 0, adaLN と出力層は 0."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.pos_embed.data.copy_(torch.from_numpy(_pos_embed_2d(self.pos_embed.shape[-1], self.grid)).float()[None])
        w = self.x_embedder.weight.data
        nn.init.xavier_uniform_(w.view(w.shape[0], -1))
        nn.init.zeros_(self.x_embedder.bias)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        for blk in self.blocks:
            nn.init.zeros_(blk.adaLN_modulation[-1].weight)
            nn.init.zeros_(blk.adaLN_modulation[-1].bias)
        for m in (self.final_adaLN[-1], self.final_linear):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def embed_context(self, upstream):
        """レーダ特徴 (B, 800, 192) → (B, 800, 384) と各ブロックの cross-attention の k, v(段をまたいで使い回せる)."""
        up = self.upstream_embedder(upstream)
        return up, [blk.crossAtten.kv(up) for blk in self.blocks]

    def forward(self, x, t, upstream_output, context=None):
        """x (B,1,24,24), t (B,) 元の 0〜999 の段, upstream_output (B, 800, 192) → ε (B,1,24,24)."""
        B = x.shape[0]
        h = self.x_embedder(x).flatten(2).transpose(1, 2) + self.pos_embed
        temb = self.t_embedder(t)
        if context is None:
            up, kvs = self.upstream_embedder(upstream_output), [None] * len(self.blocks)
        else:
            up, kvs = context
        for blk, kv in zip(self.blocks, kvs):
            h = blk(h, up, temb, kv)
        shift, scale = self.final_adaLN(temb).chunk(2, dim=1)
        h = self.final_linear(modulate(self.final_norm(h), shift, scale))       # (B, 36, 16)
        g, p = self.grid, self.p
        h = h.reshape(B, g, g, p, p, 1)
        return torch.einsum("nhwpqc->nchpwq", h).reshape(B, 1, g * p, g * p)


def _pos_embed_2d(dim: int, grid: int) -> np.ndarray:
    """公式 get_2d_sincos_pos_embed(MAE/DiT と同じ)."""
    gw, gh = np.meshgrid(np.arange(grid, dtype=np.float32), np.arange(grid, dtype=np.float32))

    def one(d, pos):
        omega = 1.0 / 10000 ** (np.arange(d // 2, dtype=np.float64) / (d / 2.0))
        out = np.einsum("m,d->md", pos.reshape(-1), omega)
        return np.concatenate([np.sin(out), np.cos(out)], axis=1)

    return np.concatenate([one(dim // 2, gw), one(dim // 2, gh)], axis=1)


def ecg_to_img(ecg: torch.Tensor) -> torch.Tensor:
    """(B, 512) → 576 に 0 詰めして (B, 1, 24, 24)(公式 patch_ecg)."""
    return F.pad(ecg, (0, ECG_IMG * ECG_IMG - ecg.shape[1])).reshape(-1, 1, ECG_IMG, ECG_IMG)


def img_to_ecg(img: torch.Tensor) -> torch.Tensor:
    return img.reshape(img.shape[0], -1)[:, :ECG_LEN]
