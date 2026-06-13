import os
import time
import json
import copy
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from diffusers import DDPMPipeline, DDIMScheduler, UNet2DModel
from tqdm import tqdm
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
import bitsandbytes as bnb

# --- Environment & Cluster Config ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "benchmark_outputs_quant2"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)

# --- Benchmark Settings ---
NUM_EVAL = 16
EVAL_BATCH = 16
GEN_BATCH = 16
TIMING_REPS = 1
TIMING_BATCHES = [1]
SAVE_BATCHES = {1}
# NUM_EVAL = 10_000
# EVAL_BATCH = 128
# GEN_BATCH = 32
# TIMING_REPS = 3
# TIMING_BATCHES = [1, 4, 8, 16, 32, 64]
# SAVE_BATCHES = {1, 4}

BASE_MODEL_ID = "google/ddpm-cifar10-32"
DDIM_STEPS = 2
FINETUNED_STEPS = 2


def load_unet(pipe, weights_path=None):
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


def get_cifar_loader():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    root = os.environ.get("CIFAR_ROOT", "./data")
    dataset = CIFAR10(root=root, train=True, download=True, transform=transform)
    return DataLoader(dataset, batch_size=EVAL_BATCH, shuffle=False, num_workers=0, pin_memory=True)


# --- Sampling Engine ---

def ddim_sample(model, sched_cfg, num_steps, batch_size, device):
    scheduler = DDIMScheduler.from_config(sched_cfg)
    scheduler.set_timesteps(num_steps)
    
    timesteps = scheduler.timesteps.to(device)
    alphas = scheduler.alphas_cumprod.to(device)
    
    x = torch.randn(batch_size, 3, 32, 32, device=device)

    for i, t in enumerate(timesteps):
        t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
        with torch.no_grad():
            noise_pred = model(x.to(next(model.parameters()).dtype), t_batch).sample.float()

        a_s = alphas[t].view(-1, 1, 1, 1)
        if i == len(timesteps) - 1:
            x = (x - torch.sqrt(1 - a_s) * noise_pred) / torch.sqrt(a_s)
            break
            
        a_e = alphas[timesteps[i + 1]].view(-1, 1, 1, 1)
        x0_p = (x - torch.sqrt(1 - a_s) * noise_pred) / torch.sqrt(a_s)
        x = torch.sqrt(a_e) * x0_p + torch.sqrt(1 - a_e) * noise_pred

    return x.clamp(-1, 1)


# --- Quantization Methods ---

def quant_int4_bnb(model):
    def _replace_linear(module):
        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                new_layer = bnb.nn.Linear4bit(
                    child.in_features, child.out_features,
                    bias=child.bias is not None, quant_type="nf4", compute_dtype=torch.float16,
                )
                new_layer.weight = bnb.nn.Params4bit(child.weight.data, requires_grad=False, quant_type="nf4")
                if child.bias is not None:
                    new_layer.bias = nn.Parameter(child.bias.data)
                setattr(module, name, new_layer)
            else:
                _replace_linear(child)

    quantized_model = copy.deepcopy(model)
    _replace_linear(quantized_model)
    return quantized_model

def quant_int4_emulated(model):
    quantized_model = copy.deepcopy(model).cpu()
    with torch.no_grad():
        for param in quantized_model.parameters():
            if param.ndim < 2:
                continue
            max_val = param.abs().amax(dim=tuple(range(1, param.ndim)), keepdim=True).clamp(min=1e-8)
            scale = max_val / 7.0
            q_int = (param / scale).round().clamp(-8, 7)
            param.copy_(q_int * scale)
    return quantized_model


# --- Metrics & Performance Evaluation ---

def get_model_size_mb(model):
    total_bytes = sum(p.nelement() * p.element_size() for p in model.parameters())
    return total_bytes / (1024 ** 2)

def evaluate_quality(name, model, sched_cfg, real_loader, steps):
    print(f"\n>>> Evaluating Quality: {name}")
    model = model.to(DEVICE).eval()

    fid_metric = FrechetInceptionDistance(feature=2048, normalize=True).to(DEVICE)
    is_metric = InceptionScore(normalize=True).to(DEVICE)

    # Process real images
    collected_reals = 0
    for imgs, _ in real_loader:
        if collected_reals >= NUM_EVAL:
            break
        imgs_01 = (imgs.to(DEVICE) + 1.0) / 2.0
        fid_metric.update(imgs_01, real=True)
        collected_reals += imgs.shape[0]

    # Generate fake images
    generated = 0
    while generated < NUM_EVAL:
        bs = min(GEN_BATCH, NUM_EVAL - generated)
        with torch.no_grad():
            samples = ddim_sample(model, sched_cfg, steps, bs, DEVICE)
        
        samples_01 = (samples.float() + 1.0) / 2.0
        fid_metric.update(samples_01, real=False)
        is_metric.update(samples_01)
        
        generated += bs
        print(f"Generated {generated}/{NUM_EVAL}", end="\r")

    fid_val = fid_metric.compute().item()
    is_mean, is_std = is_metric.compute()

    print(f"Result -> FID: {fid_val:.4f} | IS: {is_mean.item():.4f}±{is_std.item():.4f}")
    
    model.cpu()
    torch.cuda.empty_cache()
    return {"name": name, "steps": steps, "FID": round(fid_val, 4), "IS_mean": round(is_mean.item(), 4), "IS_std": round(is_std.item(), 4)}

def benchmark_speed(name, model, sched_cfg, save_tag, steps):
    print(f"Timing: {name}")
    model = model.to(DEVICE).eval()
    results = {}

    for bs in TIMING_BATCHES:
        run_times = []
        for rep in range(TIMING_REPS):
            if DEVICE == "cuda": torch.cuda.synchronize()
            t0 = time.perf_counter()
            
            with torch.no_grad():
                imgs = ddim_sample(model, sched_cfg, steps, bs, DEVICE)
                
            if DEVICE == "cuda": torch.cuda.synchronize()
            run_times.append(time.perf_counter() - t0)

            # Save sample images from the first clean run
            if rep == 0 and bs in SAVE_BATCHES:
                imgs_01 = (imgs.float() + 1.0) / 2.0
                clean_tag = save_tag.replace(" ", "_")
                vutils.save_image(imgs_01, os.path.join(OUTPUT_DIR, "images", f"{clean_tag}_bs{bs}.png"), nrow=min(bs, 4), padding=2)

        avg_time = float(np.mean(run_times))
        results[bs] = {"total_s": round(avg_time, 4), "per_image_s": round(avg_time / bs, 6)}
        print(f"   BS={bs:<3} | Total: {avg_time:.3f}s | Per image: {(avg_time / bs) * 1000:.2f} ms")

    model.cpu()
    torch.cuda.empty_cache()
    return results


# --- Variant Generation ---

def get_quant_variants(base_model, prefix, tag_prefix, steps):
    variants = [
        (f"{prefix} FP32", copy.deepcopy(base_model).cpu(), f"{tag_prefix}_fp32", steps),
        (f"{prefix} FP16", copy.deepcopy(base_model).cpu().half(), f"{tag_prefix}_fp16", steps),
        (f"{prefix} BF16", copy.deepcopy(base_model).cpu().to(torch.bfloat16), f"{tag_prefix}_bf16", steps),
        (f"{prefix} INT4-NF4", quant_int4_bnb(base_model), f"{tag_prefix}_int4_bnb", steps),
        (f"{prefix} INT4-Emu", quant_int4_emulated(base_model), f"{tag_prefix}_int4_emu", steps)
        ]
    return variants


# --- Main Orchestration ---

def main():
    pipe = DDPMPipeline.from_pretrained(BASE_MODEL_ID)
    sched_cfg = pipe.scheduler.config
    
    base_unet = load_unet(pipe)
    real_loader = get_cifar_loader()

    # Gather baseline and finetuned execution pipelines
    all_configs = get_quant_variants(base_unet, "Base", "base", DDIM_STEPS)
    ft_unet = load_unet(pipe, "fast_professor_8step.pt")
    all_configs += get_quant_variants(ft_unet, "FT-8s", "ft8s", FINETUNED_STEPS)

    # Print baseline statistics
    print("\n" + "="*50 + "\n MODEL SIZE SUMMARY \n" + "="*50)
    model_sizes = {}
    for name, model, _, steps in all_configs:
        size_mb = get_model_size_mb(model)
        params = sum(p.numel() for p in model.parameters())
        model_sizes[name] = {"size_mb": round(size_mb, 2), "params": params, "ddim_steps": steps}
        print(f"{name:<15} | Steps: {steps:<2} | Params: {params:,} | Size: {size_mb:.1f} MB")

    # Run everything
    quality_metrics = []
    timing_metrics = {}

    for name, model, save_tag, steps in all_configs:
        qual = evaluate_quality(name, model, sched_cfg, real_loader, steps)
        quality_metrics.append(qual)
        timing_metrics[name] = benchmark_speed(name, model, sched_cfg, save_tag, steps)

    # --- Print Final Clean Tables ---
    print("\n" + "="*60 + "\n FINAL BENCHMARK SUMMARY \n" + "="*60)
    for q in quality_metrics:
        print(f"{q['name']:<15} -> FID: {q['FID']:<8} | IS: {q['IS_mean']}±{q['IS_std']}")

if __name__ == "__main__":
    main()
