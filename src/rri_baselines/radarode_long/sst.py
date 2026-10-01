"""radarODE(Zhang ら, IEEE TMC 2025)の入力: 評価する 4 s 窓 [s, s+800) の中だけから作る SST と, そのエネルギー波形.

窓の外の信号は見ない(ユーザ決定). そこで試行全体ではなく, 窓ごとに SST をかける.
  1. 窓 (B, 50, 800) の各点に MATLAB wsst(解析 Morlet 'amor', 10 voices/octave)を移した変換.
     式は radarode/wsst.py と同じで, 窓をまとめて torch で計算するだけ(行は fs/N = 0.25 Hz 〜 100 Hz を対数等間隔に 80 本)
  2. 帯域の行だけを残す. 論文 p.6 "the SST plot within [1, 25] Hz"・p.9 "only contain the frequency component within
     [1, 25] Hz" に従い BAND "paper" = [1, 25] Hz を本線にする(800 点の窓では 1.06〜23.7 Hz の 42 行).
     比べる候補 "author" は著者の MMECG_to_SST.m(と CFT-RFcardi の data_generation.ipynb)が実際に取る 0.72〜100 Hz
     (試行全体 35,505 点の wsst の後半 71 行と同じ周波数範囲. 800 点の窓では 0.73〜100 Hz の 54 行).
     radarode/sst.py の 0.2〜7.7 Hz は ssqueezepy の行の向きを取り違えた誤りなので使わない(2026-10-01 訂正)
  3. 行を高い周波数から並べ(著者の load_sst の np.flip と同じ向き), 71 行へ線形補間(論文 Table I の 71)
  4. 時間方向を 200 Hz → 30 Hz(800 → 120 フレーム. 論文 p.9 "down-sampled to 30 Hz"). 区間平均で間引く
  5. 点ごとに窓の中で min-max → [0, 1](著者の sst_norm_01)
エネルギー波形(Algorithm 1 の入力, 論文 p.5 "adding the spectrogram along frequency axis")は, 2 の帯域の |SST| を
周波数方向に足した 200 Hz の波形を点ごとに [0, 1] にしたもの(間引く前の時間分解能で出す. 相手に有利な側).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

FS, NV, WIN = 200, 10, 800
N_ROWS, N_FRAMES = 71, 120
BANDS = {"paper": (1.0, 25.0), "author": (0.72, 100.0)}


def _wsst_rows(n: int = WIN, fs: float = FS, nv: int = NV) -> tuple[int, np.ndarray]:
    na = (int(np.floor(np.log2(n))) - 1) * nv
    return na, 2.0 ** np.linspace(np.log2(fs / n), np.log2(fs / 2), na)


def wsst_batch(x: torch.Tensor, fs: float = FS, nv: int = NV, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """x (..., N) 実数 → |SST| (..., na, N). radarode/wsst.py の wsst と同じ式を先頭の次元ごとにまとめて計算する.
    dtype=float64 で radarode/wsst.py と完全に一致する. 既定の float32 は速さのため(差は振幅の 1e-5 程度, 検算は __main__)."""
    lead, n = x.shape[:-1], x.shape[-1]
    x = x.reshape(-1, n).to(dtype)
    x = x - x.mean(dim=1, keepdim=True)
    na, _ = _wsst_rows(n, fs, nv)
    dev = x.device
    scales = (2.0 ** (torch.arange(1, na + 1, dtype=dtype, device=dev) / nv))[:, None]
    omega = 2 * np.pi * torch.fft.fftfreq(n, dtype=dtype, device=dev)[None]
    psi = 2.0 * torch.exp(-0.5 * (6.0 - scales * omega) ** 2) * (omega > 0)          # (na, N)
    xdft = torch.fft.fft(x)[:, None, :]                                                # (M, 1, N)
    cwt = torch.fft.ifft(xdft * psi, dim=-1)                                           # (M, na, N)
    dcwt = torch.fft.ifft(xdft * psi * (1j * omega), dim=-1)
    inst = (dcwt * cwt.conj()).imag / (cwt.real ** 2 + cwt.imag ** 2) / (2 * np.pi) * fs        # = imag(dcwt / cwt)
    del dcwt
    inst = inst.abs()
    lo, hi = np.log2(fs / n), np.log2(fs / 2)
    row = torch.floor(na / (hi - lo) * (torch.log2(inst) - lo))
    ok = torch.isfinite(row) & (row >= 0) & (row < na) & (inst >= 1e-8)              # MATLAB と同じく 1e-8 未満は捨てる
    idx = torch.where(ok, row, torch.full_like(row, na)).long()                      # 捨てる値は余分の行 na に集める
    m = x.shape[0]
    re = torch.zeros(m, na + 1, n, dtype=dtype, device=dev).scatter_add_(1, idx, cwt.real.contiguous())
    im = torch.zeros(m, na + 1, n, dtype=dtype, device=dev).scatter_add_(1, idx, cwt.imag.contiguous())
    out = torch.complex(re[:, :na], im[:, :na])
    return (out.abs() * (np.log(2) / nv)).float().reshape(*lead, na, n)


class WindowSST:
    """窓の RCG (B, 50, 800) → (SST 画像 (B, 50, 71, 120) [0, 1], エネルギー波形 (B, 50, 800) [0, 1])."""

    def __init__(self, band: str = "paper", device: str | torch.device = "cpu"):
        self.band = band
        na, freqs = _wsst_rows()
        lo, hi = BANDS[band]
        rows = np.flatnonzero((freqs >= lo) & (freqs <= hi))[::-1].copy()                # 高い方から
        self.rows = torch.as_tensor(rows, device=device)
        self.freqs = freqs[rows]
        # 行方向の線形補間(radarode/sst.py と同じ: 行番号の上で等間隔に 71 点)
        pos = np.linspace(0, len(rows) - 1, N_ROWS)
        w = np.zeros((N_ROWS, len(rows)))
        i0 = np.floor(pos).astype(int).clip(0, len(rows) - 2)
        frac = pos - i0
        w[np.arange(N_ROWS), i0] = 1 - frac
        w[np.arange(N_ROWS), i0 + 1] += frac
        self.interp = torch.as_tensor(w, dtype=torch.float32, device=device)
        self.device = device

    @torch.no_grad()
    def energy(self, rcg: torch.Tensor, chunk: int = 4) -> torch.Tensor:
        """エネルギー波形だけ (B, 50, 800) [0, 1]. Algorithm 1(PPI 推定)用."""
        return torch.cat([_minmax(wsst_batch(p.to(self.device))[:, :, self.rows].sum(dim=2), dims=(2,))
                          for p in rcg.split(chunk)])

    @torch.no_grad()
    def __call__(self, rcg: torch.Tensor, chunk: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
        imgs, energies = [], []
        for part in rcg.split(chunk):                                                    # 窓 chunk 個ずつ(装置のメモリを抑える)
            mag = wsst_batch(part.to(self.device))[:, :, self.rows]                      # (b, 50, R, 800)
            energy = mag.sum(dim=2)
            img = torch.einsum("kr,bcrt->bckt", self.interp, mag)                        # (b, 50, 71, 800)
            img = F.adaptive_avg_pool1d(img.flatten(0, 2), N_FRAMES).view(*img.shape[:3], N_FRAMES)
            imgs.append(_minmax(img, dims=(2, 3)))
            energies.append(_minmax(energy, dims=(2,)))
        return torch.cat(imgs), torch.cat(energies)


def _minmax(x: torch.Tensor, dims: tuple[int, ...]) -> torch.Tensor:
    lo, hi = x.amin(dim=dims, keepdim=True), x.amax(dim=dims, keepdim=True)
    return (x - lo) / torch.where(hi - lo < 1e-12, torch.ones_like(hi), hi - lo)


def crop_centered(img: torch.Tensor, center_frame: torch.Tensor) -> torch.Tensor:
    """窓の SST 画像 (B, 50, 71, 120) から, 拍を中心にした 120 フレームを切る. 窓の外はゼロ埋め.

    論文 p.6: "the actual SST segment is centered at the current cardiac cycle and expands to 4 seconds".
    窓の中だけを使うので, 中心が窓の端に寄った拍では窓の外が 0 になる. center_frame (B,) は整数フレーム(窓の外 ±60 まで).
    """
    padded = F.pad(img, (N_FRAMES, N_FRAMES))                                          # (B, 50, 71, 360)
    c = center_frame.long().clamp(-N_FRAMES // 2, N_FRAMES + N_FRAMES // 2 - 1)         # 窓の外に中心がある端の拍も切れる
    idx = c[:, None] + N_FRAMES // 2 + torch.arange(N_FRAMES, device=img.device)[None]
    idx = idx[:, None, None, :].expand(-1, img.shape[1], img.shape[2], -1)
    return torch.gather(padded, 3, idx)


if __name__ == "__main__":
    # 検算: 1 本の窓で radarode/wsst.py(MATLAB wsst の移植, float64)と比べる
    from rri_baselines.protocol import load_trial
    from rri_baselines.radarode.wsst import wsst
    x = torch.from_numpy(load_trial(1)["rcg_z"][1000:1800].T.copy())
    ref = wsst(x[3].double())[0].abs().float()
    for dt in (torch.float64, torch.float32):
        got = wsst_batch(x[None], dtype=dt)[0, 3]
        print(f"{dt}: 最大の差 {(ref - got).abs().max().item():.2e}(振幅の最大 {ref.max().item():.3f})")
    na, freqs = _wsst_rows()
    print(f"800 点の窓の wsst: {na} 行, {freqs[0]:.3f}〜{freqs[-1]:.1f} Hz(低い方から, log2 で等間隔. 1 行 {np.log2(freqs[1] / freqs[0]):.4f} オクターブ)")
    for band in BANDS:
        w = WindowSST(band)
        print(f"帯域 {band} {BANDS[band]} Hz: {len(w.rows)} 行(元の行番号 {int(w.rows.max())}〜{int(w.rows.min())}), "
              f"{w.freqs[0]:.2f} → {w.freqs[-1]:.2f} Hz(画像の行 0 が高い方). 71 行へ補間")
        print("   行ごとの周波数[Hz]: " + " ".join(f"{f:.2f}" for f in w.freqs))
