"""
34_generate_grids.py

Generate a 3x3 grid (9 samples) from each of the 5 models used in script 33,
saving a separate image file per model.

Models (same settings as script 33):
    base-ddpm-1000   original stochastic DDPM, 1000 steps   (weights from HF hub)
    base-ddim-25     deterministic DDIM, 25 steps           (weights from HF hub)
    student-25 / -12 / -8   distilled DDIM students at their native step counts

Usage:
    python 34_generate_grids.py --device cuda:2
    python 34_generate_grids.py --device cuda:3 --outdir grids --seed 123
    python 34_generate_grids.py --models base-ddim student   # subset by label
"""

import argparse
import functools
import os

import torch
import matplotlib
matplotlib.use("Agg")            # headless-safe: we only savefig, never show
import matplotlib.pyplot as plt

from ddpm_arch import UNet2DModel, linear_alphas_cumprod, generate_n_steps

NUM_TRAIN_TIMESTEPS = 1000


@functools.lru_cache(maxsize=1)
def fetch_base_weights():
    """Download (or reuse the cache of) google/ddpm-cifar10-32 weights from the HF hub."""
    from huggingface_hub import hf_hub_download
    for fn in ("diffusion_pytorch_model.safetensors", "diffusion_pytorch_model.bin"):
        try:
            return hf_hub_download("google/ddpm-cifar10-32", fn)
        except Exception:                                    # noqa: BLE001 (try next filename)
            continue
    raise FileNotFoundError("could not fetch google/ddpm-cifar10-32 weights from the HF hub")


# (label, weights, native_inference_steps, sampler) — same five settings as script 33.
# `weights` is a path or a callable returning one (the base is fetched lazily from HF).
MODELS = [
    ("base-ddpm-1000", fetch_base_weights, 1000, "ddpm"),
    ("base-ddim-25",   fetch_base_weights,   25, "ddim"),
    ("student-25", "./checkpoints/fast_professor_21_final.pt", 25, "ddim"),
    ("student-12", "./checkpoints/fast_professor_12step.pt",   12, "ddim"),
    ("student-8",  "./checkpoints/fast_professor_8step.pt",     8, "ddim"),
]


def load_state_dict_file(path, device):
    """Load a state dict from either a .safetensors or a .bin/.pt checkpoint."""
    if str(path).endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(path, device=str(device))
    return torch.load(path, map_location=device, weights_only=True)


def load_model(path, device):
    """Build the UNet and load a checkpoint. AttentionBlock accepts both diffusers key namings."""
    model = UNet2DModel().to(device)
    model.load_state_dict(load_state_dict_file(path, device), strict=True)
    model.eval()
    return model


@torch.no_grad()
def generate_ddpm(model, alphas_cumprod, batch_size, device, num_steps=NUM_TRAIN_TIMESTEPS):
    """Original stochastic DDPM reverse process (no DDIM shortcut)."""
    model.eval()
    ac = alphas_cumprod.to(device)
    one = torch.tensor(1.0, device=device)
    x = torch.randn(batch_size, 3, 32, 32, device=device)
    for t in range(num_steps - 1, -1, -1):
        t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
        eps = model(x, t_batch)
        a_t = ac[t]
        a_prev = ac[t - 1] if t > 0 else one
        beta_t = 1 - a_t / a_prev
        x0 = ((x - (1 - a_t).sqrt() * eps) / a_t.sqrt()).clamp(-1, 1)
        coef1 = a_prev.sqrt() * beta_t / (1 - a_t)
        coef2 = (a_t / a_prev).sqrt() * (1 - a_prev) / (1 - a_t)
        x = coef1 * x0 + coef2 * x
        if t > 0:
            x = x + beta_t.sqrt() * torch.randn_like(x)
    return x.clamp(-1, 1)


def generate(model, alphas, sampler, steps, n, device):
    """Dispatch to the DDPM or DDIM sampler. Returns n images in [-1, 1]."""
    if sampler == "ddpm":
        return generate_ddpm(model, alphas, n, device, steps)
    return generate_n_steps(model, alphas, n, device, steps)   # DDIM (from ddpm_arch)


def save_grid(images, path, title):
    """Save a 3x3 grid of images (given in [-1, 1]) to a single file."""
    imgs = ((images.clamp(-1, 1) + 1) / 2).permute(0, 2, 3, 1).cpu().numpy()
    fig, axes = plt.subplots(3, 3, figsize=(5, 5.4))
    for ax, img in zip(axes.flat, imgs):
        ax.imshow(img)
        ax.axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0,
                        help="reseeded per model, so every grid starts from the same noise")
    parser.add_argument("--outdir", default=".", help="directory for the output PNGs")
    parser.add_argument("--models", nargs="+", default=None,
                        help="only run models whose label contains one of these substrings")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    os.makedirs(args.outdir, exist_ok=True)
    print(f"Device: {device} | seed: {args.seed} | outdir: {args.outdir}")

    alphas = linear_alphas_cumprod().to(device)

    for label, weights, steps, sampler in MODELS:
        if args.models and not any(m in label for m in args.models):
            continue
        try:
            path = weights() if callable(weights) else weights   # lazily fetch base from HF
            model = load_model(path, device)
        except OSError as e:                                      # FileNotFoundError subclasses OSError
            print(f"{label:<15} skipped — weights unavailable: {e}")
            continue

        torch.manual_seed(args.seed)                 # same starting noise for every model
        images = generate(model, alphas, sampler, steps, 9, device)

        out_path = os.path.join(args.outdir, f"grid_{label}.png")
        save_grid(images, out_path, f"{label}  ({steps} steps, {sampler})")
        print(f"{label:<15} -> {out_path}")

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
