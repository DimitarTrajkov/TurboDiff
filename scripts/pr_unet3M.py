"""
pr_unet3M.py

Precision & Recall (plus FID/IS, for free) for the lightweight ~3M UNet, using
the SAME pipeline as scripts/metrics_utils.py — Inception-2048 features, k = 3 —
so the numbers line up with the distilled and LoRA tables (Appendix A.4).

eval_unet3M.py already reports FID/IS for this model but not Precision/Recall;
this script fills that gap. By default it evaluates the 50- and 20-step DDIM
configurations against 10k real CIFAR-10 images.

Usage:
    python pr_unet3M.py --device cuda:2
    python pr_unet3M.py --device cuda:2 --steps 50 20 --num-samples 10000
    python pr_unet3M.py --device cuda:2 --num-samples 2000 --data-root /path/to/cifar  # quick check
"""

import argparse
import os
import sys

import torch

import scripts.unet3M_lora as U

# metrics_utils lives in scripts/ — add it to the path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
from metrics_utils import prepare_real, compute_metrics


def load_ema_model(ckpt_path, device):
    """Load the UNet with EMA weights applied — the sampling configuration."""
    model = U.UNet().to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    ema = U.EMA(model)
    ema.load_state_dict(ckpt["ema"])
    ema.apply()
    model.eval()
    return model


@torch.no_grad()
def generate_fakes(schedule, model, steps, count, batch, device):
    """Generate `count` samples at `steps` DDIM steps; return uint8 (count,3,32,32) on CPU.

    ddim_sample already returns images in [0, 1]; compute_metrics expects uint8.
    """
    imgs, done = [], 0
    while done < count:
        bs = min(batch, count - done)
        fake01 = schedule.ddim_sample(model, (bs, 3, 32, 32), steps=steps)
        imgs.append((fake01 * 255).round().clamp(0, 255).byte().cpu())
        done += bs
        print(f"  {steps:>3}-step: generated {done}/{count}", end="\r")
    print()
    return torch.cat(imgs, dim=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint", default="checkpoints/model_final.pt")
    parser.add_argument("--steps", type=int, nargs="+", default=[50, 20])
    parser.add_argument("--num-samples", type=int, default=10000)
    parser.add_argument("--batch", type=int, default=250)
    parser.add_argument("--knn", type=int, default=3, help="k for the precision/recall k-NN manifold")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-root", default="./data",
                        help="directory holding the CIFAR-10 data (cifar-10-batches-py)")
    args = parser.parse_args()

    # The 3M sampler is pinned to the module's DEVICE global — set it before use.
    U.DEVICE = args.device
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    print(f"Device: {device} | checkpoint: {args.checkpoint} | "
          f"steps: {args.steps} | samples: {args.num_samples} | knn: {args.knn}")

    model = load_ema_model(args.checkpoint, device)
    schedule = U.DiffusionSchedule()      # reads U.DEVICE, set above

    # Real images + features, computed once and reused across step counts.
    print("Loading real images and extracting features…")
    extractor, real_imgs01, real_feats = prepare_real(args.num_samples, args.data_root, device)

    header = f"{'steps':>6}{'FID':>10}{'IS':>16}{'Precision':>12}{'Recall':>10}"
    print("\n" + header + "\n" + "-" * len(header))

    for steps in args.steps:
        torch.manual_seed(args.seed)
        fakes = generate_fakes(schedule, model, steps, args.num_samples, args.batch, device)
        m = compute_metrics(fakes, real_imgs01, real_feats, extractor, device, args.batch, args.knn)
        print(f"{steps:>6}{m['fid']:>10.3f}{m['is_mean']:>10.3f}±{m['is_std']:<5.3f}"
              f"{m['precision']:>12.3f}{m['recall']:>10.3f}")


if __name__ == "__main__":
    main()
