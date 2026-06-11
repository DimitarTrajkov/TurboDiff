"""
36_student_metrics.py

Compute FID, IS, Precision and Recall for the 8-step distilled student
(deterministic DDIM sampler) over 10k generated samples vs. 10k real CIFAR-10.

Shares the metric logic (FID / IS / Precision / Recall) with script 35 via
metrics_utils, and the architecture / DDIM sampler via ddpm_arch.

Usage:
    python 36_student_metrics.py --device cuda:2
    python 36_student_metrics.py --device cuda:2 --num-samples 2000          # quick check
    python 36_student_metrics.py --device cuda:3 --model ./checkpoints/fast_professor_12step.pt --steps 12
    python 36_student_metrics.py --device cuda:2 --data-root /path/to/cifar --out results_student8.txt
"""

import argparse

import torch
from tqdm import tqdm

from ddpm_arch import UNet2DModel, linear_alphas_cumprod, generate_n_steps
from metrics_utils import prepare_real, compute_metrics


def load_unet(path, device):
    """Load a distilled student checkpoint into the UNet (accepts .safetensors or .pt/.bin)."""
    if str(path).endswith(".safetensors"):
        from safetensors.torch import load_file
        state = load_file(path, device=str(device))
    else:
        state = torch.load(path, map_location=device, weights_only=True)
    model = UNet2DModel().to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


@torch.no_grad()
def generate_fake_images(model, alphas, steps, count, batch, device, desc):
    """Generate `count` DDIM samples at `steps` steps; return uint8 (count,3,32,32) on CPU."""
    imgs, generated = [], 0
    pbar = tqdm(total=count, desc=desc)
    while generated < count:
        bs = min(batch, count - generated)
        fake01 = (generate_n_steps(model, alphas, bs, device, steps) + 1.0) / 2.0
        imgs.append((fake01 * 255).round().byte().cpu())
        generated += bs
        pbar.update(bs)
    pbar.close()
    return torch.cat(imgs, dim=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model", default="./checkpoints/fast_professor_8step.pt",
                        help="distilled student checkpoint to evaluate")
    parser.add_argument("--steps", type=int, default=8, help="DDIM inference steps")
    parser.add_argument("--num-samples", type=int, default=10000)
    parser.add_argument("--batch", type=int, default=250)
    parser.add_argument("--knn", type=int, default=3, help="k for the precision/recall k-NN manifold")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-root", default="./data",
                        help="directory holding the CIFAR-10 data (cifar-10-batches-py)")
    parser.add_argument("--out", default=None, help="optional path to also write the results")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    print(f"Device: {device} | model: {args.model} | steps: {args.steps} | "
          f"samples: {args.num_samples} | knn: {args.knn}")

    model = load_unet(args.model, device)
    alphas = linear_alphas_cumprod().to(device)

    print("Loading real images and extracting features…")
    extractor, real_imgs01, real_feats = prepare_real(args.num_samples, args.data_root, device)

    torch.manual_seed(args.seed)
    fakes = generate_fake_images(model, alphas, args.steps, args.num_samples,
                                 args.batch, device, desc=f"{args.steps} steps")
    m = compute_metrics(fakes, real_imgs01, real_feats, extractor, device, args.batch, args.knn)

    header = f"{'steps':>6}{'FID':>10}{'IS':>16}{'Precision':>12}{'Recall':>10}"
    row = (f"{args.steps:>6}{m['fid']:>10.3f}{m['is_mean']:>10.3f}±{m['is_std']:<5.3f}"
           f"{m['precision']:>12.3f}{m['recall']:>10.3f}")
    print("\n" + header + "\n" + "-" * len(header) + "\n" + row)

    if args.out:
        with open(args.out, "w") as f:
            f.write(header + "\n" + "-" * len(header) + "\n" + row + "\n")
        print(f"\nWrote results to {args.out}")


if __name__ == "__main__":
    main()
