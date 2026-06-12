"""
Comprehensive Diffusion Model Benchmark
========================================
Evaluates: FID, IS, Precision, Recall, and Sampling Time

Models tested:
  - Distilled 8-step  (fast_professor_8step.pt)   → DDIM 8
  - Distilled 12-step (fast_professor_12step.pt)  → DDIM 12
  - Distilled 25-step (fast_professor_25step.pt)  → DDIM 25
  - Original google/ddpm-cifar10-32               → DDIM 1000, DDPM 1000, DDPM 100

Metrics:    FID, IS, Precision, Recall
Timing:     batch sizes [1, 4, 8, 16, 32, 64]
Saves:      generated images for batch sizes 1 and 4 (all configs)
"""

import os
import time
import json
import copy
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from diffusers import DDPMPipeline, DDIMScheduler, DDPMScheduler, UNet2DModel
from tqdm import tqdm
import numpy as np

# torchmetrics
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore

# ── Optional: torchmetrics Precision/Recall (needs torchmetrics >= 0.11) ──────
try:
    from torchmetrics.image import PrecisionRecallForDistributions as PRD
    HAS_PRD = True
except ImportError:
    HAS_PRD = False

# ── Optional: torch-fidelity for Precision/Recall (fallback) ──────────────────
try:
    import torch_fidelity
    HAS_FIDELITY = True
except ImportError:
    HAS_FIDELITY = False


# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
NUM_EVAL        = 10_000         # images for FID/IS/P&R
EVAL_BATCH      = 128             # batch size while feeding real images to FID
GEN_BATCH       = 100             # batch size while generating fakes for metrics
TIMING_REPS     = 3               # repeated runs per timing config (averaged)
TIMING_BATCHES  = [1, 4, 8, 16, 32, 64]
SAVE_BATCHES    = {1, 4}          # batch sizes whose outputs are saved as PNGs
OUTPUT_DIR      = "benchmark_outputs"
CIFAR_ROOT      = "./data"
BASE_MODEL_ID   = "google/ddpm-cifar10-32"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def load_base_pipeline():
    pipe = DDPMPipeline.from_pretrained(BASE_MODEL_ID)
    return pipe


def load_unet(pipe, weights_path=None):
    """Return a UNet2DModel (on CPU); optionally load custom weights."""
    model = UNet2DModel.from_config(pipe.unet.config)
    if weights_path is not None:
        state = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state)
    else:
        model.load_state_dict(pipe.unet.state_dict())
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def get_cifar_loader(batch_size=EVAL_BATCH):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = CIFAR10(root=CIFAR_ROOT, train=True, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      num_workers=0, pin_memory=True, drop_last=False)


# ─── DDIM sampler ──────────────────────────────────────────────────────────────
def ddim_sample(model, scheduler_cfg, num_steps, batch_size, device, seed=None):
    """
    Returns images in [-1, 1] as float32 tensor of shape (B, 3, 32, 32).
    Uses a deterministic DDIM trajectory.
    """
    scheduler = DDIMScheduler.from_config(scheduler_cfg)
    scheduler.set_timesteps(num_steps)
    timesteps = scheduler.timesteps.to(device)
    alphas    = scheduler.alphas_cumprod.to(device)

    if seed is not None:
        gen = torch.Generator(device=device).manual_seed(seed)
        x = torch.randn(batch_size, 3, 32, 32, device=device, generator=gen)
    else:
        x = torch.randn(batch_size, 3, 32, 32, device=device)

    for i, t in enumerate(timesteps):
        t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
        with torch.no_grad():
            noise_pred = model(x, t_batch).sample

        a_s = alphas[t].view(-1, 1, 1, 1)
        if i == len(timesteps) - 1:
            x = (x - torch.sqrt(1 - a_s) * noise_pred) / torch.sqrt(a_s)
            break

        next_t = timesteps[i + 1]
        a_e    = alphas[next_t].view(-1, 1, 1, 1)
        x0_p   = (x - torch.sqrt(1 - a_s) * noise_pred) / torch.sqrt(a_s)
        x      = torch.sqrt(a_e) * x0_p + torch.sqrt(1 - a_e) * noise_pred

    return x.clamp(-1, 1)


def ddpm_sample(model, scheduler_cfg, num_steps, batch_size, device, seed=None):
    scheduler = DDPMScheduler.from_config(scheduler_cfg)
    total = scheduler.config.num_train_timesteps  # 1000

    # Build the strided schedule explicitly
    step_ratio = total // num_steps
    timesteps = list(range(0, total, step_ratio))[::-1]  # e.g. [999, 989, ..., 9]

    alphas_cp = scheduler.alphas_cumprod.to(device)
    betas     = scheduler.betas.to(device)
    alphas    = 1.0 - betas

    if seed is not None:
        gen = torch.Generator(device=device).manual_seed(seed)
        x = torch.randn(batch_size, 3, 32, 32, device=device, generator=gen)
    else:
        x = torch.randn(batch_size, 3, 32, 32, device=device)

    for i, t_val in enumerate(timesteps):
        t_batch = torch.full((batch_size,), t_val, device=device, dtype=torch.long)
        with torch.no_grad():
            noise_pred = model(x, t_batch).sample

        a_t    = alphas_cp[t_val]
        beta_t = betas[t_val]
        alpha_t = alphas[t_val]

        x0_pred = (x - torch.sqrt(1 - a_t) * noise_pred) / torch.sqrt(a_t)
        x0_pred = x0_pred.clamp(-1, 1)

        if i < len(timesteps) - 1:
            # Previous timestep in the strided schedule (not t_val - step_ratio)
            prev_t = timesteps[i + 1]
            a_prev = alphas_cp[prev_t]

            posterior_var  = beta_t * (1 - a_prev) / (1 - a_t)
            posterior_mean = (
                torch.sqrt(a_prev) * beta_t / (1 - a_t) * x0_pred
                + torch.sqrt(alpha_t) * (1 - a_prev) / (1 - a_t) * x
            )
            x = posterior_mean + torch.sqrt(posterior_var.clamp(min=1e-20)) * torch.randn_like(x)
        else:
            x = x0_pred

    return x.clamp(-1, 1)

# ═══════════════════════════════════════════════════════════════════════════════
#  PRECISION & RECALL  (feature-space, a-la Kynkäänniemi et al.)
# ═══════════════════════════════════════════════════════════════════════════════

class PrecisionRecall:
    """
    Manifold-based Precision & Recall using Inception features.
    Reference: "Improved Precision and Recall Metric for Assessing
                Generative Models" (Kynkäänniemi et al., 2019).
    """
    def __init__(self, k=3, device="cpu"):
        self.k = k
        self.device = device
        self._build_inception()

    def _build_inception(self):
        import torchvision.models as tvm
        inc = tvm.inception_v3(weights="DEFAULT", transform_input=False)
        # Strip the classifier; keep up to the avg-pool layer
        inc.fc = torch.nn.Identity()
        inc.eval()
        self.inception = inc.to(self.device)
        self.resize = transforms.Resize((299, 299), antialias=True)

    @torch.no_grad()
    def _get_feats(self, images_01, batch_size=64):
        """images_01: tensor N×3×H×W in [0,1]"""
        feats = []
        for i in range(0, len(images_01), batch_size):
            batch = images_01[i : i + batch_size].to(self.device)
            batch = self.resize(batch)
            f = self.inception(batch)           # (B, 2048)
            feats.append(f.cpu())
        return torch.cat(feats, dim=0)          # (N, 2048)

    @staticmethod
    def _knn_precision_recall(real_feats, fake_feats, k=3):
        """
        For each fake sample: it is "precise" if it falls inside the real manifold
        (distance to its k-th real neighbour ≤ the k-th real-to-real distance).
        Symmetrically for recall.
        """
        def manifold_radii(ref, k):
            # pairwise L2 distances
            d = torch.cdist(ref, ref)                       # (N, N)
            # self-distance is 0; kth neighbour is (k+1)-th smallest
            kth, _ = d.kthvalue(k + 1, dim=1)              # (N,)
            return kth

        r_real = manifold_radii(real_feats, k)              # (Nr,)
        r_fake = manifold_radii(fake_feats, k)              # (Nf,)

        # Precision: fraction of fake that falls in real manifold
        d_f2r  = torch.cdist(fake_feats, real_feats)        # (Nf, Nr)
        min_d_f2r, _ = d_f2r.min(dim=1)                     # (Nf,)  ← nearest real
        # A fake point is in the manifold if its nearest real is within that real's radius
        nearest_real_idx = d_f2r.argmin(dim=1)
        in_real = min_d_f2r <= r_real[nearest_real_idx]
        precision = in_real.float().mean().item()

        # Recall: fraction of real that is covered by fake manifold
        d_r2f  = torch.cdist(real_feats, fake_feats)
        nearest_fake_idx = d_r2f.argmin(dim=1)
        min_d_r2f, _ = d_r2f.min(dim=1)
        in_fake = min_d_r2f <= r_fake[nearest_fake_idx]
        recall  = in_fake.float().mean().item()

        return precision, recall

    def compute(self, real_images_01, fake_images_01):
        print("  Extracting Inception features for P&R …")
        real_feats = self._get_feats(real_images_01)
        fake_feats = self._get_feats(fake_images_01)
        print("  Computing manifold P&R …")
        return self._knn_precision_recall(real_feats, fake_feats, k=self.k)


# ═══════════════════════════════════════════════════════════════════════════════
#  CORE EVALUATION ROUTINE
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_model(name, model, sampler_fn, real_loader, num_samples=NUM_EVAL):
    """t right
    Runs FID, IS, Precision, Recall for a given model/sampler combo.
    Returns dict of results.
    """
    print(f"\n{'═'*60}")
    print(f"  Evaluating: {name}")
    print(f"{'═'*60}")

    model = model.to(DEVICE)
    model.eval()

    fid_metric = FrechetInceptionDistance(feature=2048, normalize=True).to(DEVICE)
    is_metric  = InceptionScore(normalize=True).to(DEVICE)
    pr_metric  = PrecisionRecall(k=3, device=DEVICE)

    # ── 1. Collect real images ────────────────────────────────────────────────
    real_tensors = []
    real_count = 0
    with torch.no_grad():
        for imgs, _ in tqdm(real_loader, desc="  Real images → FID"):
            if real_count >= num_samples:
                break
            imgs_01 = (imgs.to(DEVICE) + 1.0) / 2.0
            fid_metric.update(imgs_01, real=True)
            real_tensors.append(imgs_01.cpu())
            real_count += imgs.shape[0]
    real_tensors = torch.cat(real_tensors, dim=0)[:num_samples]

    # ── 2. Generate fake images ───────────────────────────────────────────────
    fake_tensors = []
    fake_count = 0
    with torch.no_grad():
        while fake_count < num_samples:
            curr = min(GEN_BATCH, num_samples - fake_count)
            samples_m1_1 = sampler_fn(model, curr)          # [-1, 1]
            samples_01   = (samples_m1_1 + 1.0) / 2.0

            fid_metric.update(samples_01, real=False)
            is_metric.update(samples_01)
            fake_tensors.append(samples_01.cpu())

            fake_count += curr
            print(f"  Generated {fake_count}/{num_samples}", end="\r")
    print()
    fake_tensors = torch.cat(fake_tensors, dim=0)[:num_samples]

    # ── 3. Compute metrics ────────────────────────────────────────────────────
    fid_val   = fid_metric.compute().item()
    is_mean, is_std = is_metric.compute()
    precision, recall = pr_metric.compute(real_tensors, fake_tensors)

    results = {
        "name":      name,
        "FID":       round(fid_val, 4),
        "IS_mean":   round(is_mean.item(), 4),
        "IS_std":    round(is_std.item(), 4),
        "Precision": round(precision, 4),
        "Recall":    round(recall, 4),
    }

    print(f"\n  ┌─ {name} ─────────────────────────")
    print(f"  │  FID       : {results['FID']:.4f}")
    print(f"  │  IS        : {results['IS_mean']:.4f} ± {results['IS_std']:.4f}")
    print(f"  │  Precision : {results['Precision']:.4f}")
    print(f"  │  Recall    : {results['Recall']:.4f}")
    print(f"  └{'─'*38}")

    # ── Clean up GPU memory ───────────────────────────────────────────────────
    del fid_metric, is_metric
    model.cpu()
    torch.cuda.empty_cache()

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  TIMING BENCHMARK
# ═══════════════════════════════════════════════════════════════════════════════

def benchmark_timing(name, model, sampler_fn, save_tag):
    """
    Times the sampler for each batch size in TIMING_BATCHES.
    Saves generated grids for SAVE_BATCHES.
    Returns dict {batch_size: avg_seconds}.
    """
    print(f"\n  ⏱  Timing: {name}")
    model = model.to(DEVICE)
    model.eval()
    timing = {}

    for bs in TIMING_BATCHES:
        times = []
        for rep in range(TIMING_REPS):
            if DEVICE == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                imgs = sampler_fn(model, bs)
            if DEVICE == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append(t1 - t0)

            # Save images for the first rep of each save batch
            if rep == 0 and bs in SAVE_BATCHES:
                imgs_01 = (imgs + 1.0) / 2.0
                tag = save_tag.replace(" ", "_").replace("/", "-")
                fname = os.path.join(OUTPUT_DIR, "images",
                                     f"{tag}_bs{bs}.png")
                # Arrange in a grid (at most 4 wide)
                nrow = min(bs, 4)
                vutils.save_image(imgs_01, fname, nrow=nrow, padding=2)
                print(f"    Saved: {fname}")

        avg = float(np.mean(times))
        per_img = avg / bs
        timing[bs] = {"total_s": round(avg, 4), "per_image_s": round(per_img, 6)}
        print(f"    BS={bs:3d}  {avg:.3f}s total  ({per_img*1000:.2f} ms/img)")

    model.cpu()
    torch.cuda.empty_cache()
    return timing


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("Loading base pipeline …")
    pipe = load_base_pipeline()
    sched_cfg = pipe.scheduler.config

    # ── Load all models ────────────────────────────────────────────────────────
    print("Loading model weights …")
    orig_unet    = load_unet(pipe, weights_path=None)          # original
    unet_8step   = load_unet(pipe, "fast_professor_8step.pt")
    unet_12step  = load_unet(pipe, "fast_professor_12step.pt")
    unet_25step  = load_unet(pipe, "fast_professor_25step.pt")

    # ── Build sampler closures ─────────────────────────────────────────────────
    # Each closure: (model, batch_size) -> tensor[-1,1] on GPU
    def make_ddim(steps, unet=None):
        def sampler(model, bs):
            m = unet if unet is not None else model
            return ddim_sample(m, sched_cfg, steps, bs, DEVICE)
        return sampler

    def make_ddpm(steps):
        def sampler(model, bs):
            return ddpm_sample(model, sched_cfg, steps, bs, DEVICE)
        return sampler

    # Configuration table: (display_name, unet, sampler_fn, save_tag)
    configs = [
        ("Distilled  8-step  DDIM",   unet_8step,  make_ddim(8),           "distilled_8_ddim"),
        ("Distilled 12-step  DDIM",   unet_12step, make_ddim(12),          "distilled_12_ddim"),
        ("Distilled 25-step  DDIM",   unet_25step, make_ddim(25),          "distilled_25_ddim"),
        ("Original  8-step  DDIM",   orig_unet,  make_ddim(8),           "original_8_ddim"),
        ("Original 12-step  DDIM",   orig_unet, make_ddim(12),          "original_12_ddim"),
        ("Original 25-step  DDIM",   orig_unet, make_ddim(25),          "original_25_ddim"),
        ("Original 50-step  DDIM",   orig_unet, make_ddim(50),          "original_50_ddim"),
        ("Original  DDIM-1000",       orig_unet,   make_ddim(1000),        "original_ddim1000"),
        ("Original  DDIM-100 ",       orig_unet,   make_ddim(100),         "original_ddim100"),
        ("Original  DDPM-1000",       orig_unet,   make_ddpm(1000),        "original_ddpm1000"),
        ("Original  DDPM-100 ",       orig_unet,   make_ddpm(100),         "original_ddpm100"),

    ]

    # ── Real data loader (shared across all evaluations) ──────────────────────
    real_loader = get_cifar_loader(batch_size=EVAL_BATCH)

    all_results = []
    all_timing  = {}

    for name, unet, sampler_fn, save_tag in configs:
        # ── Quality metrics ────────────────────────────────────────────────────
        # sampler_fn already closes over unet; model arg is ignored for distilled
        # but needed for the original (we pass orig_unet directly in the closure
        # already), so the signature is uniform.
        res = evaluate_model(name=name,model=unet,real_loader=real_loader,
            sampler_fn=lambda model, bs, sf=sampler_fn: sf(model, bs))
        all_results.append(res)

        # ── Timing benchmark ───────────────────────────────────────────────────
        timing = benchmark_timing(name=name,model=unet,save_tag=save_tag,
            sampler_fn=lambda model, bs, sf=sampler_fn: sf(model, bs))
        all_timing[name] = timing

    # ═══════════════════════════════════════════════════════════════════════════
    #  PRINT SUMMARY TABLE
    # ═══════════════════════════════════════════════════════════════════════════
    col_w = 28
    print("\n\n" + "═" * 90)
    print("  FINAL SUMMARY")
    print("═" * 90)
    header = (f"{'Model':<{col_w}}  {'FID':>8}  {'IS':>10}  {'Precision':>10}  {'Recall':>8}")
    print(header)
    print("─" * 90)
    for r in all_results:
        print(
            f"{r['name']:<{col_w}}  "
            f"{r['FID']:>8.4f}  "
            f"{r['IS_mean']:>6.4f}±{r['IS_std']:<4.4f}  "
            f"{r['Precision']:>10.4f}  "
            f"{r['Recall']:>8.4f}"
        )

    print("\n\n" + "═" * 90)
    print("  SAMPLING TIME (seconds per batch)")
    print("═" * 90)
    bs_header = "  ".join([f"BS={bs:>3}" for bs in TIMING_BATCHES])
    print(f"{'Model':<{col_w}}  {bs_header}")
    print("─" * 90)
    for name, t in all_timing.items():
        times_str = "  ".join([f"{t[bs]['total_s']:>6.3f}s" for bs in TIMING_BATCHES])
        print(f"{name:<{col_w}}  {times_str}")

    print("\n\n" + "═" * 90)
    print("  SAMPLING TIME (ms per image)")
    print("═" * 90)
    print(f"{'Model':<{col_w}}  {bs_header}")
    print("─" * 90)
    for name, t in all_timing.items():
        times_str = "  ".join([f"{t[bs]['per_image_s']*1000:>7.2f}ms" for bs in TIMING_BATCHES])
        print(f"{name:<{col_w}}  {times_str}")

    # ── Save JSON report ───────────────────────────────────────────────────────
    report = {"quality_metrics": all_results, "timing": all_timing}
    report_path = os.path.join(OUTPUT_DIR, "benchmark_report2.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n✓  Full report saved to: {report_path}")
    print(f"✓  Generated images in : {os.path.join(OUTPUT_DIR, 'images')}/")


if __name__ == "__main__":
    main()
    
    
# CPU SAMPLING
# ══════════════════════════════════════════════════════════════════════════════════════════
#   SAMPLING TIME (seconds per batch)
# ══════════════════════════════════════════════════════════════════════════════════════════
# Model                         BS=  1  BS=  4  BS=  8  BS= 16  BS= 32  BS= 64
# ──────────────────────────────────────────────────────────────────────────────────────────
# Distilled  8-step  DDIM        0.981s   2.372s   4.114s   7.663s  14.298s  28.007s


# ══════════════════════════════════════════════════════════════════════════════════════════
#   SAMPLING TIME (ms per image)
# ══════════════════════════════════════════════════════════════════════════════════════════
# Model                         BS=  1  BS=  4  BS=  8  BS= 16  BS= 32  BS= 64
# ──────────────────────────────────────────────────────────────────────────────────────────
# Distilled  8-step  DDIM        981.32ms   592.89ms   514.23ms   478.95ms   446.83ms   437.61ms
