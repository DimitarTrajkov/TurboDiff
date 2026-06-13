import argparse
import copy
import math
import time
from xml.parsers.expat import model
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore


LORA_BANDS = [(0,249),(250,499),(500,749),(750,999)]
N_LORA_BANDS = len(LORA_BANDS)

from scripts.unet3M_lora import DEVICE, DiffusionSchedule, UNet, EMA




def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)

    ac = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2

    ac = ac / ac[0]

    betas = 1 - ac[1:] / ac[:-1]
    return betas.clamp(1e-4, 0.9999)


@torch.no_grad()
def benchmark_sampling_time(model, schedule, batch_size=32, ddim_steps=50, trials=5):
    """
    Benchmarks the time taken to sample a specific batch size of images.
    Calculates average, mean (same as avg), and median times over multiple trials.
    """
    print(f"\nRunning timing benchmark over {trials} trials for batch size {batch_size}...")
    durations = []
    
    # Warmup loop to clear lazy initializations / CUDA contexts
    _ = schedule.ddim_sample(model, (batch_size, 3, 32, 32), steps=ddim_steps)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    for t in range(trials):
        start_time = time.perf_counter()
        
        _ = schedule.ddim_sample(model, (batch_size, 3, 32, 32), steps=ddim_steps)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        end_time = time.perf_counter()
        durations.append(end_time - start_time)
        # print(f" Trial {t+1}: {durations[-1]:.4f} seconds")

    avg_time = np.mean(durations)
    median_time = np.median(durations)
    
    print(f"--- Timing Results CPU({batch_size} batch) ---")
    print(f"Average (Mean) Time: {avg_time:.4f} seconds")
    # print(f"Median Time:         {median_time:.4f} seconds")
    # print("----------------------")
    print(f"{((avg_time / batch_size) * 1000):.2f}ms per img\n")
    return avg_time, median_time


@torch.no_grad()
def evaluate(ckpt_path, n_generated=10000, batch_size=256, ddim_steps=20):

    model = UNet().to(DEVICE)
    ema = EMA(model)

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    ema.load_state_dict(ckpt["ema"])
    ema.apply()

    schedule = DiffusionSchedule()

    # Benchmark the sampling performance for 32 images first
    for batch in [32, 64, 128]:
        benchmark_sampling_time(model, schedule, batch_size=batch, ddim_steps=ddim_steps, trials=1)

    # Initialize standard metrics
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(DEVICE)
    is_metric = InceptionScore(normalize=True).to(DEVICE)

    real_tf = transforms.ToTensor()
    real_ds = datasets.CIFAR10("./data", train=True, download=False, transform=real_tf)
    real_loader = DataLoader(real_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    print("Loading real images...")
    count = 0
    for imgs, _ in real_loader:
        imgs = imgs.to(DEVICE)
        imgs_uint8 = (imgs * 255).to(torch.uint8)
        
        fid.update(imgs_uint8, real=True)

        count += len(imgs)
        if count >= n_generated:
            break

    print("Generating fake images...")
    remaining = n_generated

    while remaining > 0:
        cur_bs = min(batch_size, remaining)
        fake = schedule.ddim_sample(model, (cur_bs, 3, 32, 32), steps=ddim_steps)
        fake_uint8 = (fake * 255).clamp(0, 255).to(torch.uint8)

        fid.update(fake_uint8, real=False)
        is_metric.update(fake_uint8)

        remaining -= cur_bs
        print(f"{n_generated-remaining}/{n_generated}")

    fid_score = fid.compute().item()
    is_mean, is_std = is_metric.compute()

    print("\n====================")
    print(f"FID:       {fid_score:.4f}")
    print(f"IS :       {is_mean:.4f} +- {is_std:.4f}")
    print("====================")


if __name__ == "__main__":
    evaluate("model_final.pt",10_000, 32, 50)
    
# 50 steps DDIM
# ====================
# FID: 15.8078
# IS : 5.3641 ± 0.1333
# ====================


# 20 steps DDIM
# ====================
# FID: 19.3013
# IS : 5.2486 ± 0.1098
# ==================== 
# --- Timing Results (batch 32)---
# Average (Mean) Time: 0.3928 seconds
# Median Time:         0.3941 seconds
# ----------------------
# 12.28ms per img




# 50 steps DDIM GPU Timing:
# (32 batch) -- 0.9278s / 28.99ms per img
# (64 batch) -- 1.7590s / 27.48ms per img
# (128 batch) -- 3.5563s / 27.78ms per img
# (256 batch) -- 7.1152s / 27.79ms per img

# 50 steps DDIM CPU Timing:
# CPU(32 batch) -- 22.0643s / 689.51ms per img
# CPU(64 batch) -- 42.5926s / 665.51ms per img
# CPU(128 batch) -- 85.6463s / 669.11ms per img
