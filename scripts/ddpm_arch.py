"""
ddpm_arch.py

Shared building blocks for the pure-PyTorch DDPM / DDIM scripts (29–32).

Covers:
  - Full UNet2DModel matching google/ddpm-cifar10-32 state-dict key names
  - linear_alphas_cumprod()   – scheduler maths
  - make_training_grid()      – DDIM timestep grid used during distillation
  - ddim_step()               – one DDIM update (used in both training and eval)
  - generate_n_steps()        – evaluation sampling loop
  - run_eval()                – FID + IS evaluation
"""

import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torch.utils.data import DataLoader
from tqdm import tqdm

WEIGHTS_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--google--ddpm-cifar10-32"
    "/snapshots/267b167dc01f0e4e61923ea244e8b988f84deb80"
    "/diffusion_pytorch_model.bin"
)


# ─────────────────────────────────────────────────────────────
# 1.  SINUSOIDAL TIME EMBEDDING
# ─────────────────────────────────────────────────────────────
def sinusoidal_embedding(timesteps: torch.Tensor, dim: int,
                          downscale_freq_shift: float = 1.0) -> torch.Tensor:
    """Matches diffusers get_timestep_embedding (flip_sin_to_cos=False, freq_shift=1)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000)
        * torch.arange(half, dtype=torch.float32, device=timesteps.device)
        / (half - downscale_freq_shift)
    )
    args = timesteps[:, None].float() * freqs[None]
    return torch.cat([args.sin(), args.cos()], dim=-1)   # (B, dim)


class TimestepEmbedding(nn.Module):
    def __init__(self, in_ch: int, embed_dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_ch, embed_dim)
        self.act      = nn.SiLU()
        self.linear_2 = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        return self.linear_2(self.act(self.linear_1(x)))


# ─────────────────────────────────────────────────────────────
# 2.  RESNET BLOCK
# ─────────────────────────────────────────────────────────────
class ResnetBlock2D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, temb_ch: int,
                 groups: int = 32, eps: float = 1e-6):
        super().__init__()
        self.norm1         = nn.GroupNorm(groups, in_ch,  eps=eps, affine=True)
        self.conv1         = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_emb_proj = nn.Linear(temb_ch, out_ch)
        self.norm2         = nn.GroupNorm(groups, out_ch, eps=eps, affine=True)
        self.conv2         = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.nonlinearity  = nn.SiLU()
        self.conv_shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x, temb):
        h = self.conv1(self.nonlinearity(self.norm1(x)))
        h = h + self.time_emb_proj(self.nonlinearity(temb))[:, :, None, None]
        h = self.conv2(self.nonlinearity(self.norm2(h)))
        if self.conv_shortcut is not None:
            x = self.conv_shortcut(x)
        return x + h


# ─────────────────────────────────────────────────────────────
# 3.  ATTENTION BLOCK
#     Key names: group_norm / query / key / value / proj_attn
# ─────────────────────────────────────────────────────────────
class AttentionBlock(nn.Module):
    # Newer diffusers renamed the attention projections. Accept both on load so
    # checkpoints saved by either version (legacy google .bin or recently trained
    # students) match this module under strict=True.
    _KEY_ALIASES = {"to_q": "query", "to_k": "key", "to_v": "value", "to_out.0": "proj_attn"}

    def __init__(self, ch: int, groups: int = 32, eps: float = 1e-6):
        super().__init__()
        self.group_norm = nn.GroupNorm(groups, ch, eps=eps, affine=True)
        self.query      = nn.Linear(ch, ch)
        self.key        = nn.Linear(ch, ch)
        self.value      = nn.Linear(ch, ch)
        self.proj_attn  = nn.Linear(ch, ch)
        self.scale      = ch ** -0.5

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Rename any modern-diffusers keys to this module's names before loading.
        for alt, own in self._KEY_ALIASES.items():
            for suffix in ("weight", "bias"):
                alt_key, own_key = prefix + alt + "." + suffix, prefix + own + "." + suffix
                if alt_key in state_dict and own_key not in state_dict:
                    state_dict[own_key] = state_dict.pop(alt_key)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.group_norm(x).view(B, C, -1).transpose(1, 2)   # (B, N, C)
        q, k, v = self.query(h), self.key(h), self.value(h)
        attn = torch.softmax(torch.bmm(q, k.transpose(1, 2)) * self.scale, dim=-1)
        h    = self.proj_attn(torch.bmm(attn, v))
        return x + h.transpose(1, 2).view(B, C, H, W)


# ─────────────────────────────────────────────────────────────
# 4.  DOWNSAMPLER / UPSAMPLER
# ─────────────────────────────────────────────────────────────
class Downsample2D(nn.Module):
    """Asymmetric padding (downsample_padding=0 in config)."""
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=0)

    def forward(self, x):
        return self.conv(F.pad(x, (0, 1, 0, 1)))


class Upsample2D(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))


# ─────────────────────────────────────────────────────────────
# 5.  DOWN BLOCKS
# ─────────────────────────────────────────────────────────────
class DownBlock2D(nn.Module):
    def __init__(self, in_ch, out_ch, temb_ch, num_layers=2, add_downsample=True):
        super().__init__()
        self.resnets = nn.ModuleList([
            ResnetBlock2D(in_ch if i == 0 else out_ch, out_ch, temb_ch)
            for i in range(num_layers)
        ])
        self.downsamplers = nn.ModuleList([Downsample2D(out_ch)]) if add_downsample else None

    def forward(self, x, temb):
        skips = ()
        for r in self.resnets:
            x = r(x, temb);  skips += (x,)
        if self.downsamplers:
            for d in self.downsamplers:
                x = d(x)
            skips += (x,)
        return x, skips


class AttnDownBlock2D(nn.Module):
    def __init__(self, in_ch, out_ch, temb_ch, num_layers=2, add_downsample=True):
        super().__init__()
        self.resnets = nn.ModuleList([
            ResnetBlock2D(in_ch if i == 0 else out_ch, out_ch, temb_ch)
            for i in range(num_layers)
        ])
        self.attentions  = nn.ModuleList([AttentionBlock(out_ch) for _ in range(num_layers)])
        self.downsamplers = nn.ModuleList([Downsample2D(out_ch)]) if add_downsample else None

    def forward(self, x, temb):
        skips = ()
        for r, a in zip(self.resnets, self.attentions):
            x = a(r(x, temb));  skips += (x,)
        if self.downsamplers:
            for d in self.downsamplers:
                x = d(x)
            skips += (x,)
        return x, skips


# ─────────────────────────────────────────────────────────────
# 6.  MID BLOCK
# ─────────────────────────────────────────────────────────────
class UNetMidBlock2D(nn.Module):
    def __init__(self, ch, temb_ch):
        super().__init__()
        self.resnets    = nn.ModuleList([ResnetBlock2D(ch, ch, temb_ch),
                                         ResnetBlock2D(ch, ch, temb_ch)])
        self.attentions = nn.ModuleList([AttentionBlock(ch)])

    def forward(self, x, temb):
        x = self.attentions[0](self.resnets[0](x, temb))
        return self.resnets[1](x, temb)


# ─────────────────────────────────────────────────────────────
# 7.  UP BLOCKS
#     resnet_specs: list of (in_ch, out_ch) per layer.
#     in_ch already encodes the skip-cat size (e.g., 512 = 256+256).
# ─────────────────────────────────────────────────────────────
class UpBlock2D(nn.Module):
    def __init__(self, resnet_specs, temb_ch, add_upsample=True):
        super().__init__()
        self.resnets    = nn.ModuleList([ResnetBlock2D(ic, oc, temb_ch) for ic, oc in resnet_specs])
        self.upsamplers = nn.ModuleList([Upsample2D(resnet_specs[-1][1])]) if add_upsample else None

    def forward(self, x, temb, skip_stack):
        for r in self.resnets:
            x = r(torch.cat([x, skip_stack.pop()], dim=1), temb)
        if self.upsamplers:
            for u in self.upsamplers:
                x = u(x)
        return x


class AttnUpBlock2D(nn.Module):
    def __init__(self, resnet_specs, temb_ch, add_upsample=True):
        super().__init__()
        out_ch          = resnet_specs[0][1]
        self.resnets    = nn.ModuleList([ResnetBlock2D(ic, oc, temb_ch) for ic, oc in resnet_specs])
        self.attentions = nn.ModuleList([AttentionBlock(out_ch) for _ in resnet_specs])
        self.upsamplers = nn.ModuleList([Upsample2D(out_ch)]) if add_upsample else None

    def forward(self, x, temb, skip_stack):
        for r, a in zip(self.resnets, self.attentions):
            x = a(r(torch.cat([x, skip_stack.pop()], dim=1), temb))
        if self.upsamplers:
            for u in self.upsamplers:
                x = u(x)
        return x


# ─────────────────────────────────────────────────────────────
# 8.  FULL UNET  (matches google/ddpm-cifar10-32 exactly)
# ─────────────────────────────────────────────────────────────
class UNet2DModel(nn.Module):
    """
    Pure-PyTorch UNet matching google/ddpm-cifar10-32.

    Config: block_out_channels=[128,256,256,256], layers_per_block=2
    down: [DownBlock2D, AttnDownBlock2D, DownBlock2D, DownBlock2D]
    up  : [UpBlock2D,   UpBlock2D,       AttnUpBlock2D, UpBlock2D]

    forward(x, t) → noise prediction tensor   (no .sample wrapper)
    """
    def __init__(self):
        super().__init__()
        temb_dim = 512

        self.conv_in        = nn.Conv2d(3, 128, 3, padding=1)
        self.time_embedding = TimestepEmbedding(128, temb_dim)

        self.down_blocks = nn.ModuleList([
            DownBlock2D    (128, 128, temb_dim, num_layers=2, add_downsample=True),
            AttnDownBlock2D(128, 256, temb_dim, num_layers=2, add_downsample=True),
            DownBlock2D    (256, 256, temb_dim, num_layers=2, add_downsample=True),
            DownBlock2D    (256, 256, temb_dim, num_layers=2, add_downsample=False),
        ])

        self.mid_block = UNetMidBlock2D(256, temb_dim)

        self.up_blocks = nn.ModuleList([
            UpBlock2D    ([(512,256),(512,256),(512,256)], temb_dim, add_upsample=True),
            UpBlock2D    ([(512,256),(512,256),(512,256)], temb_dim, add_upsample=True),
            AttnUpBlock2D([(512,256),(512,256),(384,256)], temb_dim, add_upsample=True),
            UpBlock2D    ([(384,128),(256,128),(256,128)], temb_dim, add_upsample=False),
        ])

        self.conv_norm_out = nn.GroupNorm(32, 128, eps=1e-6, affine=True)
        self.conv_act      = nn.SiLU()
        self.conv_out      = nn.Conv2d(128, 3, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        temb = sinusoidal_embedding(t, 128, downscale_freq_shift=1.0)
        temb = self.time_embedding(temb)

        x = self.conv_in(x)

        skip_stack = [x]
        for block in self.down_blocks:
            x, skips = block(x, temb)
            skip_stack.extend(skips)

        x = self.mid_block(x, temb)

        for block in self.up_blocks:
            x = block(x, temb, skip_stack)

        return self.conv_out(self.conv_act(self.conv_norm_out(x)))


# ─────────────────────────────────────────────────────────────
# 9.  SCHEDULER HELPERS
# ─────────────────────────────────────────────────────────────
def linear_alphas_cumprod(beta_start: float = 0.0001, beta_end: float = 0.02,
                           num_steps: int = 1000) -> torch.Tensor:
    """Returns alpha_bar_t = cumprod(1 - beta_t) for a linear beta schedule."""
    betas = torch.linspace(beta_start, beta_end, num_steps, dtype=torch.float64)
    return torch.cumprod(1.0 - betas, dim=0).float()


def make_training_grid(n_steps: int, num_train: int = 1000) -> torch.Tensor:
    """
    DDIM training timestep grid with n_steps evenly spaced values,
    in descending order (from noisy → clean).
    Matches diffusers DDIMScheduler.set_timesteps(n_steps).
    """
    step = num_train // n_steps
    ts   = (np.arange(0, n_steps) * step).round().astype(np.int64)[::-1].copy()
    return torch.from_numpy(ts)


def ddim_step(eps: torch.Tensor, x_t: torch.Tensor,
              a_s: torch.Tensor, a_e: torch.Tensor) -> torch.Tensor:
    """
    One deterministic DDIM step (eta=0).
    eps : predicted noise  (B, C, H, W)
    x_t : current noisy sample
    a_s : alpha_bar at current timestep   – shape broadcastable to (B,1,1,1)
    a_e : alpha_bar at previous timestep  – shape broadcastable to (B,1,1,1)
    Returns x_{t-1}.
    """
    x0 = (x_t - (1 - a_s).sqrt() * eps) / a_s.sqrt()
    return a_e.sqrt() * x0 + (1 - a_e).sqrt() * eps


# ─────────────────────────────────────────────────────────────
# 10.  GENERATION & EVALUATION
# ─────────────────────────────────────────────────────────────
@torch.no_grad()
def generate_n_steps(model: UNet2DModel, alphas_cumprod: torch.Tensor,
                     batch_size: int, device: str, n_steps: int) -> torch.Tensor:
    """
    Generate a batch using n_steps linearly spaced DDIM steps.
    Matches the sampling loop in scripts 21, 22, 27.
    """
    model.eval()
    x = torch.randn(batch_size, 3, 32, 32, device=device)
    timesteps = torch.linspace(999, 0, n_steps, dtype=torch.long, device=device)
    alphas    = alphas_cumprod.to(device)

    for i, t in enumerate(timesteps):
        t_val   = t.item()
        t_batch = torch.full((batch_size,), t_val, device=device, dtype=torch.long)
        eps     = model(x, t_batch)
        a_s     = alphas[t_val].view(1, 1, 1, 1)

        if i == len(timesteps) - 1:
            x = (x - (1 - a_s).sqrt() * eps) / a_s.sqrt()
        else:
            a_e = alphas[timesteps[i + 1].item()].view(1, 1, 1, 1)
            x   = ddim_step(eps, x, a_s, a_e)

    return x.clamp(-1, 1)


def run_eval(model: UNet2DModel, alphas_cumprod: torch.Tensor,
             dataset, device: str,
             num_samples: int = 10000, steps: int = 25,
             gen_batch: int = 64) -> None:
    """
    Compute FID and IS over num_samples generated images.
    dataset should be the torchvision CIFAR-10 dataset (normalized to [-1,1]).
    """
    print(f"\n--- Evaluating {num_samples} samples at {steps} steps ---")
    fid       = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    is_metric = InceptionScore(normalize=True).to(device)

    # real images
    real_loader = DataLoader(dataset, batch_size=128, shuffle=False)
    real_count  = 0
    with torch.no_grad():
        for imgs, _ in tqdm(real_loader, desc="Real images", leave=False):
            if real_count >= num_samples:
                break
            batch = ((imgs[:num_samples - real_count].to(device) + 1.0) / 2.0)
            fid.update(batch, real=True)
            real_count += batch.shape[0]

    # fake images
    fake_count = 0
    with torch.no_grad():
        while fake_count < num_samples:
            bs      = min(gen_batch, num_samples - fake_count)
            samples = generate_n_steps(model, alphas_cumprod, bs, device, steps)
            samples = (samples + 1.0) / 2.0
            fid.update(samples, real=False)
            is_metric.update(samples)
            fake_count += bs
            print(f"  generated {fake_count}/{num_samples}", end="\r")

    print(f"\nFID: {fid.compute().item():.4f} | IS: {is_metric.compute()[0].item():.4f}")
