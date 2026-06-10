"""
29_pytorch_only_ddpm.py

Replicates script 14 (DDPM 1000 steps vs DDIM 30 steps on google/ddpm-cifar10-32)
using only PyTorch — no diffusers library.

Architecture and helpers are imported from ddpm_arch.py.
"""

import math
import time
import numpy as np
import torch
import matplotlib.pyplot as plt

from scripts.ddpm_arch import (
    UNet2DModel,
    linear_alphas_cumprod,
    WEIGHTS_PATH,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED   = 42
BATCH  = 64


# ─────────────────────────────────────────────
# SCHEDULERS (specific to the 1000-step comparison)
# ─────────────────────────────────────────────
class DDPMScheduler:
    """Stochastic DDPM reverse process. variance_type=fixed_large, clip_sample=True."""
    def __init__(self):
        self.alphas_cumprod = linear_alphas_cumprod()
        self.timesteps      = torch.arange(999, -1, -1)

    @torch.no_grad()
    def step(self, eps_pred: torch.Tensor, t: int, x_t: torch.Tensor) -> torch.Tensor:
        a_t    = self.alphas_cumprod[t].to(x_t.device)
        a_prev = self.alphas_cumprod[t - 1].to(x_t.device) if t > 0 else torch.tensor(1.0)
        beta_t = 1 - a_t / a_prev

        x0     = ((x_t - (1 - a_t).sqrt() * eps_pred) / a_t.sqrt()).clamp(-1, 1)
        coef1  = a_prev.sqrt() * beta_t / (1 - a_t)
        coef2  = (a_t / a_prev).sqrt() * (1 - a_prev) / (1 - a_t)
        mean   = coef1 * x0 + coef2 * x_t

        if t > 0:
            mean = mean + beta_t.sqrt() * torch.randn_like(x_t)
        return mean


class DDIMScheduler:
    """Deterministic DDIM (eta=0), same linear betas."""
    def __init__(self, num_inference_steps: int = 30):
        self.alphas_cumprod      = linear_alphas_cumprod()
        self.num_train_timesteps = 1000
        self.set_timesteps(num_inference_steps)

    def set_timesteps(self, n: int):
        self.num_inference_steps = n
        step        = self.num_train_timesteps // n
        self._step  = step
        ts          = (np.arange(0, n) * step).round().astype(np.int64)[::-1].copy()
        self.timesteps = torch.from_numpy(ts)

    @torch.no_grad()
    def step(self, eps_pred: torch.Tensor, t: int, x_t: torch.Tensor) -> torch.Tensor:
        a_t    = self.alphas_cumprod[t].to(x_t.device)
        prev_t = t - self._step
        a_prev = self.alphas_cumprod[prev_t].to(x_t.device) if prev_t >= 0 else torch.tensor(1.0)
        x0     = (x_t - (1 - a_t).sqrt() * eps_pred) / a_t.sqrt()
        return a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * eps_pred


# ─────────────────────────────────────────────
# GENERATION HELPERS
# ─────────────────────────────────────────────
def save_grid(tensor: torch.Tensor, path: str, title: str):
    imgs = ((tensor.clamp(-1, 1) + 1) / 2).permute(0, 2, 3, 1).cpu().numpy()
    n    = int(math.sqrt(len(imgs)))
    fig, axes = plt.subplots(n, n, figsize=(n, n))
    for ax, img in zip(axes.flat, imgs):
        ax.imshow(img);  ax.axis("off")
    fig.suptitle(title, fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  => saved {path}")


@torch.no_grad()
def run_ddpm(model, scheduler, batch, device):
    torch.manual_seed(SEED)
    x  = torch.randn(batch, 3, 32, 32, device=device)
    t0 = time.time()
    for t in scheduler.timesteps:
        t_b = torch.full((batch,), t.item(), device=device, dtype=torch.long)
        x   = scheduler.step(model(x, t_b), t.item(), x)
    return x, time.time() - t0


@torch.no_grad()
def run_ddim(model, scheduler, batch, device):
    torch.manual_seed(SEED)
    x  = torch.randn(batch, 3, 32, 32, device=device)
    t0 = time.time()
    for t in scheduler.timesteps:
        t_b = torch.full((batch,), t.item(), device=device, dtype=torch.long)
        x   = scheduler.step(model(x, t_b), t.item(), x)
    return x, time.time() - t0


if __name__ == "__main__":
    print(f"Device: {DEVICE}")

    print("Loading weights …")
    model = UNet2DModel().to(DEVICE)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE, weights_only=True), strict=True)
    model.eval()
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M")

    # TEST 1 – DDPM 1000 steps
    print("\n" + "=" * 40)
    print(" TEST 1: DDPM (1000 steps)")
    print("=" * 40)
    imgs_ddpm, ddpm_time = run_ddpm(model, DDPMScheduler(), BATCH, DEVICE)
    print(f"  Time: {ddpm_time:.2f}s")
    save_grid(imgs_ddpm, "29_ddpm_1000_steps.png", f"DDPM 1000 steps ({ddpm_time:.2f}s)")

    # TEST 2 – DDIM 30 steps
    print("\n" + "=" * 40)
    print(" TEST 2: DDIM (30 steps)")
    print("=" * 40)
    imgs_ddim, ddim_time = run_ddim(model, DDIMScheduler(30), BATCH, DEVICE)
    print(f"  Time: {ddim_time:.2f}s")
    save_grid(imgs_ddim, "29_ddim_30_steps.png", f"DDIM 30 steps ({ddim_time:.2f}s)")

    print("\n" + "=" * 40)
    print(f"  DDPM 1000 steps : {ddpm_time:.2f}s")
    print(f"  DDIM  30  steps : {ddim_time:.2f}s")
    print(f"  Speedup         : {ddpm_time / ddim_time:.1f}x faster")
