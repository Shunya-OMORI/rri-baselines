"""mm-JEPA-ECG の拡散(公式 diffusion/ @ 4423ef6 の書き直し. 元は OpenAI guided-diffusion → DiT).

- 学習: 線形 β(1e-4〜0.02)・1000 段(論文 p.38967, 公式 create_diffusion(timestep_respacing=""))
        ε 予測の MSE(式 (15)). 分散は固定(learn_sigma=False → FIXED_LARGE. DDIM では使わない)
- 推論: DDIM 100 段・η=0(公式 inference_from_train.py の create_diffusion(timestep_respacing="100") と
        ddim_sample_loop(eta=0.0), clip_denoised=True)
"""
import numpy as np
import torch

T_TRAIN = 1000


def linear_betas(n: int = T_TRAIN) -> np.ndarray:
    scale = 1000 / n
    return np.linspace(scale * 1e-4, scale * 0.02, n, dtype=np.float64)


def space_timesteps(num_timesteps: int, count: int) -> list[int]:
    """公式 respace.space_timesteps(num, "100") と同じ: 区間 1 つを等間隔に count 点(端を含む, 丸め)."""
    if count <= 1:
        return [0]
    frac = (num_timesteps - 1) / (count - 1)
    return sorted({int(round(i * frac)) for i in range(count)})


class Diffusion:
    """use_steps を与えると公式 SpacedDiffusion と同じく β を詰め直し, モデルには元の段番号を渡す."""

    def __init__(self, n_steps: int = T_TRAIN, respace: int | None = None):
        base = np.cumprod(1.0 - linear_betas(n_steps))
        use = list(range(n_steps)) if respace is None else space_timesteps(n_steps, respace)
        last, betas = 1.0, []
        for i in use:
            betas.append(1 - base[i] / last)
            last = base[i]
        self.timestep_map = torch.tensor(use, dtype=torch.long)
        self.num_timesteps = len(use)
        ac = np.cumprod(1.0 - np.array(betas))
        self.alphas_cumprod = torch.tensor(ac, dtype=torch.float64)
        self.alphas_cumprod_prev = torch.tensor(np.append(1.0, ac[:-1]), dtype=torch.float64)

    def _at(self, arr: torch.Tensor, t: torch.Tensor, ndim: int) -> torch.Tensor:
        return arr.to(t.device)[t].float().view(-1, *([1] * (ndim - 1)))

    def q_sample(self, x0, t, noise):
        a = self._at(self.alphas_cumprod, t, x0.ndim)
        return a.sqrt() * x0 + (1 - a).sqrt() * noise

    def model_t(self, t: torch.Tensor) -> torch.Tensor:
        return self.timestep_map.to(t.device)[t]

    def pred_xstart(self, x_t, t, eps, clip: bool = True):
        """公式 p_mean_variance の pred_xstart(clip_denoised=True の既定で [-1, 1] に切る)."""
        a = self._at(self.alphas_cumprod, t, x_t.ndim)
        x0 = (1 / a).sqrt() * x_t - (1 / a - 1).sqrt() * eps
        return x0.clamp(-1, 1) if clip else x0

    @torch.no_grad()
    def ddim_sample(self, model, noise, clip: bool = True, **model_kwargs):
        """公式 ddim_sample_loop(eta=0.0). model(x, 元の段, **kwargs) → ε."""
        x = noise
        for i in reversed(range(self.num_timesteps)):
            t = torch.full((x.shape[0],), i, device=x.device, dtype=torch.long)
            eps_hat = model(x, self.model_t(t), **model_kwargs)
            x0 = self.pred_xstart(x, t, eps_hat, clip)
            a = self._at(self.alphas_cumprod, t, x.ndim)
            a_prev = self._at(self.alphas_cumprod_prev, t, x.ndim)
            eps = ((1 / a).sqrt() * x - x0) / (1 / a - 1).sqrt()       # 公式 _predict_eps_from_xstart
            x = x0 * a_prev.sqrt() + (1 - a_prev).sqrt() * eps
        return x
