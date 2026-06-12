"""
Fast CIFAR-10 Diffusion — scratch training, no teacher.

  • Small UNet (~3M params)
  • Per-noise-level LoRA on time projections (4 bands)
  • Cosine noise schedule
  • DDIM sampler at inference (20 steps default)
  • EMA of weights for sampling
"""

import argparse
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────────
# Global config
# ──────────────────────────────────────────────────────────────────────────────

DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
T         = 1000          # total diffusion timesteps
IMG_SIZE  = 32
CHANNELS  = 3

# LoRA bands — each gets its own low-rank adapter on the time projection
LORA_BANDS   = [(0, 249), (250, 499), (500, 749), (750, 999)]
N_LORA_BANDS = len(LORA_BANDS)


def get_band(t: torch.Tensor) -> torch.Tensor:
    """Map timestep → band index (long tensor, same device as t)."""
    band = torch.zeros_like(t)
    for i, (lo, hi) in enumerate(LORA_BANDS):
        band = torch.where((t >= lo) & (t <= hi), torch.full_like(t, i), band)
    return band


# ──────────────────────────────────────────────────────────────────────────────
# Noise schedule
# ──────────────────────────────────────────────────────────────────────────────

def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    ac = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    ac = ac / ac[0]
    betas = 1 - ac[1:] / ac[:-1]
    return betas.clamp(1e-4, 0.9999)


class DiffusionSchedule:
    def __init__(self):
        betas     = cosine_beta_schedule(T).to(DEVICE)
        alphas    = 1.0 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        self.betas                    = betas
        self.alphas_bar               = alphas_bar
        self.sqrt_ab                  = alphas_bar.sqrt()
        self.sqrt_one_minus_ab        = (1 - alphas_bar).sqrt()

    # ── forward process ──────────────────────────────────────────────────────

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor,
                 noise: torch.Tensor | None = None):
        """x_t = sqrt(ᾱ_t) x0 + sqrt(1−ᾱ_t) ε"""
        if noise is None:
            noise = torch.randn_like(x0)
        s  = self.sqrt_ab[t].view(-1, 1, 1, 1)
        sm = self.sqrt_one_minus_ab[t].view(-1, 1, 1, 1)
        return s * x0 + sm * noise, noise

    # ── DDIM sampler ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def ddim_sample(self, model, shape: tuple, steps: int = 20,
                    eta: float = 0.0) -> torch.Tensor:
        """
        Deterministic DDIM (eta=0) or stochastic (eta=1 ≈ DDPM).
        `steps` can be far smaller than T — 20 works well.
        """
        model.eval()
        b  = shape[0]
        x  = torch.randn(shape, device=DEVICE)
        ts = torch.linspace(T - 1, 0, steps, dtype=torch.long, device=DEVICE)

        for i, t_cur in enumerate(ts):
            t_b  = t_cur.expand(b)
            band = get_band(t_b)

            ab_cur  = self.alphas_bar[t_cur]
            ab_prev = self.alphas_bar[ts[i + 1]] if i + 1 < len(ts) else \
                      torch.tensor(1.0, device=DEVICE)

            eps     = model(x, t_b, band)
            x0_pred = ((x - self.sqrt_one_minus_ab[t_cur] * eps) /
                       self.sqrt_ab[t_cur]).clamp(-1, 1)

            sigma   = eta * ((1 - ab_prev) / (1 - ab_cur) *
                             (1 - ab_cur / ab_prev)).sqrt()
            dir_xt  = (1 - ab_prev - sigma ** 2).sqrt() * eps
            noise   = sigma * torch.randn_like(x) if eta > 0 else 0.0

            x = ab_prev.sqrt() * x0_pred + dir_xt + noise

        return (x.clamp(-1, 1) + 1) / 2   # → [0, 1]


# ──────────────────────────────────────────────────────────────────────────────
# LoRA linear layer  (one (A,B) pair per noise band)
# ──────────────────────────────────────────────────────────────────────────────

class LoRALinear(nn.Module):
    """
    W_eff = W_base + (1/r) · B_i · A_i   for band i

    Each band gets independent A ∈ R^{r×in}, B ∈ R^{out×r}.
    B is initialised to zero → identity at init, no training instability.
    """

    def __init__(self, in_f: int, out_f: int, rank: int = 8,
                 bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_f, out_f, bias=bias)
        self.rank   = rank
        self.scale  = 1.0 / rank

        self.lora_A = nn.Parameter(
            torch.randn(N_LORA_BANDS, rank, in_f) * 0.01)
        self.lora_B = nn.Parameter(
            torch.zeros(N_LORA_BANDS, out_f, rank))

    def forward(self, x: torch.Tensor, band: torch.Tensor) -> torch.Tensor:
        # x: [B, in_f]   band: [B]
        base = self.linear(x)                        # [B, out_f]
        A    = self.lora_A[band]                     # [B, rank, in_f]
        B    = self.lora_B[band]                     # [B, out_f, rank]
        delta = torch.bmm(B, torch.bmm(A, x.unsqueeze(-1))).squeeze(-1)
        return base + self.scale * delta


# ──────────────────────────────────────────────────────────────────────────────
# Building blocks
# ──────────────────────────────────────────────────────────────────────────────

class SinusoidalPE(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half  = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) *
            torch.arange(half, device=t.device, dtype=torch.float) / (half - 1)
        )
        args = t[:, None].float() * freqs[None]
        return torch.cat([args.sin(), args.cos()], dim=-1)


def safe_group_norm(n_groups: int, channels: int) -> nn.GroupNorm:
    """Pick the largest divisor of `channels` that is <= n_groups."""
    g = next(g for g in range(n_groups, 0, -1) if channels % g == 0)
    return nn.GroupNorm(g, channels)


class ResBlock(nn.Module):
    """
    Standard GroupNorm + Conv residual block.
    Time conditioning via scale-shift (FiLM).
    The time projection is a LoRALinear — band-specific adaptation at zero
    extra cost during inference (just pick the right (A,B) pair).
    """

    def __init__(self, in_ch: int, out_ch: int, time_dim: int,
                 lora_rank: int = 8, n_groups: int = 8):
        super().__init__()
        self.norm1    = safe_group_norm(n_groups, in_ch)
        self.conv1    = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2    = safe_group_norm(n_groups, out_ch)
        self.conv2    = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_mlp = LoRALinear(time_dim, out_ch * 2, rank=lora_rank)
        self.skip     = (nn.Conv2d(in_ch, out_ch, 1)
                         if in_ch != out_ch else nn.Identity())

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor,
                band: torch.Tensor) -> torch.Tensor:
        h             = self.conv1(F.silu(self.norm1(x)))
        scale, shift  = self.time_mlp(t_emb, band).chunk(2, dim=-1)
        h             = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        h             = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x, *_):
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x, *_):
        return self.op(F.interpolate(x, scale_factor=2, mode="nearest"))


class SelfAttention(nn.Module):
    """Lightweight single-head self-attention for the bottleneck."""

    def __init__(self, ch: int):
        super().__init__()
        self.norm = safe_group_norm(8, ch)
        self.qkv  = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)
        self.scale = ch ** -0.5

    def forward(self, x):
        B, C, H, W = x.shape
        h   = self.norm(x)
        qkv = self.qkv(h).view(B, 3, C, H * W)
        q, k, v = qkv.unbind(dim=1)
        attn = torch.softmax(torch.bmm(q.transpose(1, 2), k) * self.scale, dim=-1)
        out  = torch.bmm(v, attn.transpose(1, 2)).view(B, C, H, W)
        return x + self.proj(out)


# ──────────────────────────────────────────────────────────────────────────────
# UNet
# ──────────────────────────────────────────────────────────────────────────────

class UNet(nn.Module):
    """
    ~3M parameter UNet.
    ch_mult=(1,2,2): channels are [64, 128, 128] across 3 resolutions.
    Self-attention at the bottleneck helps global coherence cheaply.
    """

    def __init__(self, in_ch: int = CHANNELS, base_ch: int = 64,
                 ch_mult: tuple = (1, 2, 2), time_dim: int = 128,
                 lora_rank: int = 8):
        super().__init__()
        chs = [base_ch * m for m in ch_mult]

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPE(time_dim),
            nn.Linear(time_dim, time_dim * 4), nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )

        self.init_conv = nn.Conv2d(in_ch, chs[0], 3, padding=1)

        # Encoder
        self.enc_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        ch = chs[0]
        for ch_out in chs[1:]:
            self.enc_blocks.append(ResBlock(ch, ch_out, time_dim, lora_rank))
            self.downs.append(Downsample(ch_out))
            ch = ch_out

        # Bottleneck (res → attention → res)
        self.mid_res1 = ResBlock(ch, ch, time_dim, lora_rank)
        self.mid_attn = SelfAttention(ch)
        self.mid_res2 = ResBlock(ch, ch, time_dim, lora_rank)

        # Decoder — skip channels must mirror what the encoder actually outputs.
        # Encoder iterates chs[1:], so skips have those channel counts.
        # Reversing chs[:-1] is wrong when ch_mult has repeated values.
        # Correct: reverse chs[1:] to get the true skip channel sequence.
        self.dec_blocks = nn.ModuleList()
        self.ups        = nn.ModuleList()
        for ch_skip in reversed(chs[1:]):
            self.ups.append(Upsample(ch))
            self.dec_blocks.append(ResBlock(ch + ch_skip, ch_skip, time_dim, lora_rank))
            ch = ch_skip

        # After the decoder loop, ch == chs[1] (last skip channel = 128).
        # out_res projects back down to chs[0]=64, then out projects to in_ch.
        self.out_res = ResBlock(ch, chs[0], time_dim, lora_rank)
        self.out = nn.Sequential(safe_group_norm(8, chs[0]),nn.SiLU(),nn.Conv2d(chs[0], in_ch, 1),)

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                band: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(t)
        h     = self.init_conv(x)

        skips = []
        for res, down in zip(self.enc_blocks, self.downs):
            h = res(h, t_emb, band)
            skips.append(h)
            h = down(h)

        h = self.mid_res1(h, t_emb, band)
        h = self.mid_attn(h)
        h = self.mid_res2(h, t_emb, band)

        for up, res, skip in zip(self.ups, self.dec_blocks, reversed(skips)):
            h = up(h)
            h = torch.cat([h, skip], dim=1)
            h = res(h, t_emb, band)

        h = self.out_res(h, t_emb, band)
        return self.out(h)


# ──────────────────────────────────────────────────────────────────────────────
# EMA helper
# ──────────────────────────────────────────────────────────────────────────────

class EMA:
    """
    Exponential moving average of model weights.
    Use ema.apply() before sampling, ema.restore() to resume training.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.model  = model
        self.decay  = decay
        self.shadow = copy.deepcopy(model.state_dict())

    @torch.no_grad()
    def update(self):
        for k, v in self.model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k] = self.decay * self.shadow[k] + (1 - self.decay) * v

    def apply(self):
        """Swap EMA weights into the model (for sampling)."""
        self._backup = copy.deepcopy(self.model.state_dict())
        self.model.load_state_dict(self.shadow)

    def restore(self):
        """Restore training weights."""
        self.model.load_state_dict(self._backup)

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, sd):
        self.shadow = sd


# ──────────────────────────────────────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────────────────────────────────────

def get_loader(batch_size: int = 128) -> DataLoader:    
    tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),   # → [-1, 1]
    ])
    ds = datasets.CIFAR10("./data", train=True, download=True, transform=tf)
    return DataLoader(ds, batch_size=batch_size, shuffle=True,num_workers=4, pin_memory=True, drop_last=True)


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def train(epochs = 500, lr: float = 2e-4, batch_size = 128, lora_rank = 8, save_every = 50, save_path = "model_final.pt"):

    print("═" * 60)
    print(f"Training UNet from scratch  |  device={DEVICE}")
    print(f"  lora_rank={lora_rank}  epochs={epochs}  batch={batch_size}")
    print("═" * 60)

    schedule = DiffusionSchedule()
    model    = UNet(lora_rank=lora_rank).to(DEVICE)
    ema      = EMA(model)
    loader   = get_loader(batch_size)

    opt     = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs * len(loader), eta_min=lr / 10)
    scaler  = torch.amp.GradScaler("cuda", enabled=(DEVICE == "cuda"))


    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        bar = tqdm(loader, desc=f"Epoch {epoch:>4}/{epochs}")

        for x, _ in bar:
            x    = x.to(DEVICE)
            t    = torch.randint(0, T, (x.size(0),), device=DEVICE)
            band = get_band(t)

            with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
                x_t, noise = schedule.q_sample(x, t)
                pred       = model(x_t, t, band)
                loss       = F.mse_loss(pred, noise)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            lr_sched.step()
            ema.update()

            total_loss += loss.item()
            bar.set_postfix(loss=f"{loss.item():.4f}",
                            lr=f"{lr_sched.get_last_lr()[0]:.2e}")

        avg = total_loss / len(loader)
        print(f"  ↳ Epoch {epoch}  avg loss: {avg:.4f}")

        if epoch % save_every == 0 or epoch == epochs:
            ckpt = {
                "epoch":    epoch,
                "model":    model.state_dict(),
                "ema":      ema.state_dict(),
                "opt":      opt.state_dict(),
                "lr_sched": lr_sched.state_dict(),
            }
            path = save_path if epoch == epochs else f"model_e{epoch}.pt"
            torch.save(ckpt, path)
            print(f"  ↳ Saved → {path}")

            # Quick sample grid every save
            _quick_sample(model, ema, schedule, tag=f"e{epoch}")

    print(f"✓ Done — final checkpoint: {save_path}")


def _quick_sample(model, ema, schedule, n=16, steps=20, tag=""):
    ema.apply()
    imgs = schedule.ddim_sample(model, (n, CHANNELS, IMG_SIZE, IMG_SIZE), steps=steps)
    save_image(imgs, f"sample_{tag}.png", nrow=4)
    ema.restore()
    model.train()

if __name__ == "__main__":
    train(epochs=500, lr=2e-4, batch_size=128, lora_rank=8, save_every=50,resume=None, save_path="model_final.pt")


