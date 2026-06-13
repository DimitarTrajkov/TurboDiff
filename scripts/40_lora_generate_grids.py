"""
40_lora_generate_grids.py

Generate a 3x3 grid (9 samples) from each per-step LoRA student trained by
script 37, saving one image file per model — the LoRA counterpart of script 34.

LoRA checkpoints carry a bank of per-step heads and need head switching during
sampling, so this script drives them through script 37's loader
(`load_lora_student`) and sampler (`generate_lora`), imported directly.

Usage:
    python 40_lora_generate_grids.py --device cuda:2 \
        --checkpoints checkpoints_v2/lora_student_25step.pt \
                      checkpoints_v2/lora_student_12step.pt \
                      checkpoints_v3/lora_student_8step.pt
    python 40_lora_generate_grids.py --device cuda:3 --outdir grids --seed 123 \
        --checkpoints checkpoints_v3/lora_student_8step.pt
"""

import argparse
import importlib.util
import os

import torch
import matplotlib
matplotlib.use("Agg")            # headless-safe: we only savefig, never show
import matplotlib.pyplot as plt

from ddpm_arch import linear_alphas_cumprod


def _import_script37():
    """Import the LoRA machinery from 37_lora_progressive_distillation.py (digit-prefixed)."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "37_lora_progressive_distillation.py")
    spec = importlib.util.spec_from_file_location("lora37", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


L = _import_script37()


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
    parser.add_argument("--checkpoints", nargs="+", required=True,
                        help="LoRA student checkpoints from script 37")
    parser.add_argument("--seed", type=int, default=100,
                        help="reseeded per model, so every grid starts from the same noise")
    parser.add_argument("--outdir", default=".", help="directory for the output PNGs")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    os.makedirs(args.outdir, exist_ok=True)
    print(f"Device: {device} | seed: {args.seed} | outdir: {args.outdir}")

    alphas = linear_alphas_cumprod().to(device)

    for ckpt in args.checkpoints:
        if not os.path.exists(ckpt):
            print(f"{os.path.basename(ckpt):<28} skipped — file not found: {ckpt}")
            continue
        model, schedule = L.load_lora_student(ckpt, device)
        model.eval()
        steps = len(schedule)
        label = os.path.basename(ckpt).replace(".pt", "")

        torch.manual_seed(args.seed)                 # same starting noise for every model
        images = L.generate_lora(model, schedule, alphas, 9, device)

        out_path = os.path.join(args.outdir, f"grid_{label}.png")
        save_grid(images, out_path, f"{label}  ({steps} steps, LoRA)")
        print(f"{label:<28} -> {out_path}")

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
