"""
benchmark_unet3M.py

For the lightweight ~3M UNet (trained by unet3M_lora.py):
  - save a 3x3 sample grid          (as in script 34)
  - run a precise inference-time benchmark (as in script 33: CUDA-event timing,
    per-(batch) warm-up, median + p90, ms/img, img/s, peak memory)

It loads the EMA weights from the checkpoint (the sampling configuration, same as
eval_unet3M.py), and samples with the model's own DDIM sampler (cosine schedule,
per-band LoRA), which returns images already in [0, 1].

FID/IS for this model live in eval_unet3M.py; this script focuses on the grid and
inference timing.

Usage:
    python benchmark_unet3M.py --device cuda:2
    python benchmark_unet3M.py --device cuda:2 --steps 20 --outdir grids
    python benchmark_unet3M.py --device cuda:2 --skip-grid --batch-sizes 32 64 128 256
    python benchmark_unet3M.py --device cuda:2 --steps 50 --skip-benchmark   # grid only
"""

import argparse
import os
import time

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")            # headless-safe: we only savefig, never show
import matplotlib.pyplot as plt

import scripts.unet3M_lora as U

DEFAULT_BATCH_SIZES = [1, 4, 8, 16, 32, 64]


def load_ema_model(ckpt_path, device):
    """Load the UNet with EMA weights applied — the sampling configuration."""
    model = U.UNet().to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    ema = U.EMA(model)
    ema.load_state_dict(ckpt["ema"])
    ema.apply()                      # swap EMA weights into the model for sampling
    model.eval()
    return model


def save_grid(images01, path, title):
    """images01: (9,3,32,32) in [0,1] (the range ddim_sample returns)."""
    imgs = images01.clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
    fig, axes = plt.subplots(3, 3, figsize=(5, 5.4))
    for ax, img in zip(axes.flat, imgs):
        ax.imshow(img)
        ax.axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def time_config(schedule, model, batch, steps, device, warmup, timed):
    """Latency stats (ms) for one batch size, timing the full DDIM sampling call."""
    use_cuda = device.type == "cuda"
    if use_cuda:
        torch.cuda.reset_peak_memory_stats(device)

    def run():
        return schedule.ddim_sample(model, (batch, 3, 32, 32), steps=steps)

    for _ in range(warmup):
        run()
    if use_cuda:
        torch.cuda.synchronize(device)

    times_ms = []
    for _ in range(timed):
        if use_cuda:
            torch.cuda.synchronize(device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            run()
            end.record()
            torch.cuda.synchronize(device)
            times_ms.append(start.elapsed_time(end))
        else:
            t0 = time.perf_counter()
            run()
            times_ms.append((time.perf_counter() - t0) * 1000.0)

    times_ms = np.array(times_ms)
    p50 = float(np.median(times_ms))
    peak_mb = (torch.cuda.max_memory_allocated(device) / 1e6) if use_cuda else float("nan")
    return {
        "p50_ms": p50,
        "p90_ms": float(np.percentile(times_ms, 90)),
        "ms_per_img": p50 / batch,
        "img_per_s": batch / (p50 / 1000.0),
        "peak_mb": peak_mb,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint", default="checkpoints/model_final.pt")
    parser.add_argument("--steps", type=int, default=50, help="DDIM sampling steps")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=DEFAULT_BATCH_SIZES)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--timed", type=int, default=20)
    parser.add_argument("--outdir", default=".", help="directory for the grid PNG")
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--skip-grid", action="store_true")
    parser.add_argument("--skip-benchmark", action="store_true")
    args = parser.parse_args()

    # The 3M sampler is pinned to the module's DEVICE global — set it before use.
    U.DEVICE = args.device
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f"Device: {device} ({torch.cuda.get_device_name(device)}) | steps: {args.steps}")
    else:
        print(f"Device: {device} (CPU — perf_counter timing) | steps: {args.steps}")

    model = load_ema_model(args.checkpoint, device)
    schedule = U.DiffusionSchedule()      # reads U.DEVICE, set above
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded {args.checkpoint} | {n_params / 1e6:.2f}M params")

    # ── Grid (as in script 34) ────────────────────────────────────────────────
    if not args.skip_grid:
        os.makedirs(args.outdir, exist_ok=True)
        torch.manual_seed(args.seed)
        imgs = schedule.ddim_sample(model, (9, 3, 32, 32), steps=args.steps)   # [0,1]
        out_path = os.path.join(args.outdir, f"grid_unet3M_{args.steps}step.png")
        save_grid(imgs, out_path, f"unet3M  ({args.steps} steps, DDIM)")
        print(f"grid -> {out_path}")

    # ── Inference-time benchmark (as in script 33) ────────────────────────────
    if not args.skip_benchmark:
        header = (f"{'model':<10}{'steps':>6}{'batch':>7}{'p50 ms':>10}{'p90 ms':>10}"
                  f"{'ms/img':>9}{'img/s':>10}{'peak MB':>10}")
        print("\n--- Inference time ---\n" + header + "\n" + "-" * len(header))
        for batch in args.batch_sizes:
            try:
                s = time_config(schedule, model, batch, args.steps, device,
                                args.warmup, args.timed)
            except RuntimeError as e:
                if "out of memory" not in str(e).lower():
                    raise
                print(f"{'unet3M':<10}{args.steps:>6}{batch:>7}  (OOM — skipped)")
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                continue
            print(f"{'unet3M':<10}{args.steps:>6}{batch:>7}{s['p50_ms']:>10.2f}"
                  f"{s['p90_ms']:>10.2f}{s['ms_per_img']:>9.3f}{s['img_per_s']:>10.1f}"
                  f"{s['peak_mb']:>10.1f}")


if __name__ == "__main__":
    main()
