"""
35_ddpm_step_metrics.py

Evaluate the base google/ddpm-cifar10-32 model with the ORIGINAL stochastic DDPM
sampler (no DDIM) at several step counts, computing FID, IS, Precision and Recall
over 10k generated samples vs. 10k real CIFAR-10 images.

Fewer-step DDPM uses a respaced ancestral schedule (a strided subsequence of the
1000 training timesteps, posterior recomputed for that subsequence). It is still
stochastic — fresh noise is injected at every step — unlike DDIM.

FID and IS reuse the same torchmetrics setup as ddpm_arch.run_eval / script 09
(feature=2048, normalize=True). Precision/Recall are the improved manifold metric
(Kynkaanniemi et al. 2019) computed on the same Inception-2048 features as FID.

Usage:
    python 35_ddpm_step_metrics.py --device cuda:2
    python 35_ddpm_step_metrics.py --device cuda:3 --steps 1000 100 8 --num-samples 10000
    python 35_ddpm_step_metrics.py --device cuda:2 --num-samples 2000   # quick sanity run

WARNING: the 1000-step setting does 1000 forward passes per image. At 10k samples
that is ~10M forward passes and can take hours. Use --num-samples to scope a quick
check first.
"""

import argparse
import functools
import glob
import os

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from tqdm import tqdm

from ddpm_arch import UNet2DModel, linear_alphas_cumprod

NUM_TRAIN_TIMESTEPS = 1000


# ─────────────────────────────────────────────────────────────
# Model loading (base weights fetched from the HF hub)
# ─────────────────────────────────────────────────────────────
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


def load_base_model(device):
    path = fetch_base_weights()
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        state = load_file(path, device=str(device))
    else:
        state = torch.load(path, map_location=device, weights_only=True)
    model = UNet2DModel().to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────
# Stochastic DDPM sampler with respacing (no DDIM)
# ─────────────────────────────────────────────────────────────
@torch.no_grad()
def generate_ddpm_steps(model, alphas_cumprod, batch_size, device, num_steps):
    """
    Ancestral DDPM sampling over a strided subsequence of the training timesteps.
    num_steps == 1000 reproduces the full original DDPM chain (999..0).
    """
    model.eval()
    ac = alphas_cumprod.to(device)
    one = torch.tensor(1.0, device=device)
    ts = torch.linspace(NUM_TRAIN_TIMESTEPS - 1, 0, num_steps, device=device).round().long()
    ts = [int(t) for t in ts]                                # strided, descending to 0

    x = torch.randn(batch_size, 3, 32, 32, device=device)
    for i, t in enumerate(ts):
        eps = model(x, torch.full((batch_size,), t, device=device, dtype=torch.long))
        a_t = ac[t]
        a_prev = ac[ts[i + 1]] if i < num_steps - 1 else one
        beta = 1 - a_t / a_prev
        x0 = ((x - (1 - a_t).sqrt() * eps) / a_t.sqrt()).clamp(-1, 1)
        coef1 = a_prev.sqrt() * beta / (1 - a_t)
        coef2 = (a_t / a_prev).sqrt() * (1 - a_prev) / (1 - a_t)
        x = coef1 * x0 + coef2 * x
        if i < num_steps - 1:                                # stochastic: add noise except final step
            x = x + beta.sqrt() * torch.randn_like(x)
    return x.clamp(-1, 1)


# ─────────────────────────────────────────────────────────────
# Inception features + improved Precision / Recall
# ─────────────────────────────────────────────────────────────
@torch.no_grad()
def inception_features(extractor, imgs01, device, chunk=250):
    """Extract Inception-2048 features (same network FID uses) for images in [0, 1]."""
    feats = []
    for i in range(0, imgs01.shape[0], chunk):
        batch = imgs01[i:i + chunk].to(device)
        feats.append(extractor((batch * 255).byte()).float())
    return torch.cat(feats, dim=0)


@torch.no_grad()
def _kth_nn_radius(feats, k, chunk=2000):
    """Distance from each point to its k-th nearest neighbour within the same set."""
    n = feats.shape[0]
    radii = torch.empty(n, device=feats.device)
    for i in range(0, n, chunk):
        d = torch.cdist(feats[i:i + chunk], feats)           # (c, n)
        # exclude self (distance 0): the (k+1)-th smallest is the k-th neighbour
        radii[i:i + chunk] = torch.topk(d, k + 1, largest=False).values[:, -1]
    return radii


@torch.no_grad()
def _fraction_within(query, ref, ref_radii, chunk=2000):
    """Fraction of query points lying inside ANY reference hypersphere."""
    n = query.shape[0]
    inside = torch.zeros(n, dtype=torch.bool, device=query.device)
    for i in range(0, n, chunk):
        d = torch.cdist(query[i:i + chunk], ref)             # (c, m)
        inside[i:i + chunk] = (d <= ref_radii.unsqueeze(0)).any(dim=1)
    return inside.float().mean().item()


@torch.no_grad()
def precision_recall(real_feats, fake_feats, k=3):
    """
    Improved precision & recall (Kynkaanniemi et al. 2019).
      precision = fraction of fakes inside the real feature manifold
      recall    = fraction of reals inside the fake feature manifold
    """
    real_radii = _kth_nn_radius(real_feats, k)
    fake_radii = _kth_nn_radius(fake_feats, k)
    precision = _fraction_within(fake_feats, real_feats, real_radii)
    recall = _fraction_within(real_feats, fake_feats, fake_radii)
    return precision, recall


# ─────────────────────────────────────────────────────────────
# Real data
# ─────────────────────────────────────────────────────────────
def load_real_images(num_samples):
    """First num_samples CIFAR-10 train images as a CPU tensor in [0, 1]."""
    dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transforms.ToTensor())
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    imgs, seen = [], 0
    for x, _ in loader:
        imgs.append(x)
        seen += x.shape[0]
        if seen >= num_samples:
            break
    return torch.cat(imgs, dim=0)[:num_samples]


# ─────────────────────────────────────────────────────────────
# Evaluation for one step count
# ─────────────────────────────────────────────────────────────
@torch.no_grad()
def generate_fake_images(model, alphas, steps, count, batch, device, desc):
    """Generate `count` DDPM samples at `steps` steps; return uint8 (count,3,32,32) on CPU."""
    imgs, generated = [], 0
    pbar = tqdm(total=count, desc=desc)
    while generated < count:
        bs = min(batch, count - generated)
        fake01 = (generate_ddpm_steps(model, alphas, bs, device, steps) + 1.0) / 2.0
        imgs.append((fake01 * 255).round().byte().cpu())
        generated += bs
        pbar.update(bs)
    pbar.close()
    return torch.cat(imgs, dim=0)


@torch.no_grad()
def compute_metrics(fake_uint8, real_imgs01, real_feats, extractor, device, batch, knn):
    """FID / IS / Precision / Recall for a set of fake images (uint8, NCHW)."""
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    is_metric = InceptionScore(normalize=True).to(device)

    for i in range(0, real_imgs01.shape[0], batch):
        fid.update(real_imgs01[i:i + batch].to(device), real=True)

    fake_feats = []
    for i in range(0, fake_uint8.shape[0], batch):
        fake01 = fake_uint8[i:i + batch].to(device).float() / 255.0
        fid.update(fake01, real=False)
        is_metric.update(fake01)
        fake_feats.append(inception_features(extractor, fake01, device))
    fake_feats = torch.cat(fake_feats, dim=0)

    precision, recall = precision_recall(real_feats, fake_feats, k=knn)
    is_mean, is_std = is_metric.compute()
    return {"fid": fid.compute().item(), "is_mean": is_mean.item(),
            "is_std": is_std.item(), "precision": precision, "recall": recall}


def load_fake_shards(dirpath, steps, num_samples):
    """Concatenate all saved shards for a step count into one uint8 tensor."""
    files = sorted(glob.glob(os.path.join(dirpath, f"fakes_steps{steps}_shard*.pt")))
    if not files:
        raise FileNotFoundError(f"no shards for steps={steps} in {dirpath}")
    fakes = torch.cat([torch.load(f, map_location="cpu", weights_only=True) for f in files], dim=0)
    if fakes.shape[0] < num_samples:
        print(f"  warning: only {fakes.shape[0]} samples for steps={steps} (<{num_samples})")
    return fakes[:num_samples]


def shard_count(total, shard_id, num_shards):
    """How many samples this shard generates so the shards sum to `total`."""
    return total // num_shards + (1 if shard_id < total % num_shards else 0)


def generate_shard(args, device):
    """Generation-only mode: produce this process's shard of fakes and save to disk."""
    shard_id, num_shards = (0, 1)
    if args.shard:
        shard_id, num_shards = (int(p) for p in args.shard.split("/"))
    os.makedirs(args.save_fakes, exist_ok=True)
    torch.manual_seed(args.seed + shard_id)              # distinct, reproducible samples per shard

    model = load_base_model(device)
    alphas = linear_alphas_cumprod().to(device)
    for steps in args.steps:
        count = shard_count(args.num_samples, shard_id, num_shards)
        fakes = generate_fake_images(model, alphas, steps, count, args.batch, device,
                                     desc=f"steps{steps} shard{shard_id}/{num_shards}")
        path = os.path.join(args.save_fakes, f"fakes_steps{steps}_shard{shard_id}of{num_shards}.pt")
        torch.save(fakes, path)
        print(f"  saved {tuple(fakes.shape)} -> {path}")


def run_metrics(args, device):
    """Metric mode: fakes come from disk (--load-fakes) or are generated on this GPU."""
    print("Loading real images and extracting features…")
    real_imgs01 = load_real_images(args.num_samples)
    extractor = FrechetInceptionDistance(feature=2048, normalize=True).to(device).inception
    real_feats = inception_features(extractor, real_imgs01, device)

    model = alphas = None
    if not args.load_fakes:
        model = load_base_model(device)
        alphas = linear_alphas_cumprod().to(device)
        torch.manual_seed(args.seed)

    header = f"{'steps':>6}{'FID':>10}{'IS':>16}{'Precision':>12}{'Recall':>10}"
    lines = [header, "-" * len(header)]
    print("\n" + header + "\n" + "-" * len(header))

    for steps in args.steps:
        if args.load_fakes:
            fakes = load_fake_shards(args.load_fakes, steps, args.num_samples)
        else:
            fakes = generate_fake_images(model, alphas, steps, args.num_samples,
                                         args.batch, device, desc=f"{steps:>4} steps")
        m = compute_metrics(fakes, real_imgs01, real_feats, extractor, device, args.batch, args.knn)
        row = (f"{steps:>6}{m['fid']:>10.3f}{m['is_mean']:>10.3f}±{m['is_std']:<5.3f}"
               f"{m['precision']:>12.3f}{m['recall']:>10.3f}")
        print(row)
        lines.append(row)

    if args.out:
        with open(args.out, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"\nWrote results to {args.out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, nargs="+", default=[1000, 100, 8])
    parser.add_argument("--num-samples", type=int, default=10000)
    parser.add_argument("--batch", type=int, default=250)
    parser.add_argument("--knn", type=int, default=3, help="k for the precision/recall k-NN manifold")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None, help="optional path to also write the results table")
    # Multi-GPU sharding: each GPU generates a shard, then one process aggregates.
    parser.add_argument("--shard", default=None,
                        help="this process's shard as i/N (e.g. 0/4); use with --save-fakes")
    parser.add_argument("--save-fakes", default=None,
                        help="generation-only: save this shard's fakes to this directory")
    parser.add_argument("--load-fakes", default=None,
                        help="skip generation: load fakes from this directory and compute metrics")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    print(f"Device: {device} | samples: {args.num_samples} | steps: {args.steps} | knn: {args.knn}")

    if args.save_fakes:
        generate_shard(args, device)
    else:
        run_metrics(args, device)


if __name__ == "__main__":
    main()
