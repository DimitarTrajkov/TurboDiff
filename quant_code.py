"""
quantization_benchmark.py  (CINECA/Leonardo offline-safe version)
─────────────────────────────────────────────────────────────────
Identical to the original, with two changes needed on HPC clusters
where compute nodes have no internet access:

  1. load_base_pipeline() passes local_files_only=True so it never
     tries to contact huggingface.co (avoids OfflineModeIsEnabled).
  2. The HF_HOME env-var is read at startup; if it is not set the
     script falls back to a sensible scratch default and warns you.
"""

import os, time, json, copy, math
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from diffusers import DDPMPipeline, DDIMScheduler, DDPMScheduler, UNet2DModel
from tqdm import tqdm
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore

# ── Optional bitsandbytes for true INT4 / INT8 CUDA kernels ──────────────────
try:
    import bitsandbytes as bnb
    HAS_BNB = True
except ImportError:
    HAS_BNB = False

# ─────────────────────────────────────────────────────────────────────────────
#  HF CACHE  – resolve before any HF import side-effects
# ─────────────────────────────────────────────────────────────────────────────
_DEFAULT_HF_HOME = (
    "/leonardo_scratch/large/userexternal/dtraykov/hf_cache"  # ← adjust if needed
)
if "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = _DEFAULT_HF_HOME
    print(f"[WARN] HF_HOME not set — using default: {_DEFAULT_HF_HOME}")
else:
    print(f"[config] HF_HOME={os.environ['HF_HOME']}")

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG  (mirrors original script)
# ─────────────────────────────────────────────────────────────────────────────
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
NUM_EVAL        = 10_000
EVAL_BATCH      = 128
GEN_BATCH       = 32
TIMING_REPS     = 3
TIMING_BATCHES  = [1, 4, 8, 16, 32, 64]
SAVE_BATCHES    = {1, 4}
OUTPUT_DIR      = "benchmark_outputs_quant"
CIFAR_ROOT      = os.environ.get("CIFAR_ROOT", "./data")
BASE_MODEL_ID   = "google/ddpm-cifar10-32"
DDIM_STEPS      = 50          # step count for base-model configs
FINETUNED_CKPT  = os.environ.get(
    "FINETUNED_CKPT",
    "/leonardo/home/userexternal/dtraykov/fast_professor_8step.pt"
)
FINETUNED_STEPS = 8           # step count for finetuned-model configs

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)
print(f"[config] device={DEVICE}  CIFAR_ROOT={CIFAR_ROOT}")


# ─────────────────────────────────────────────────────────────────────────────
#  PIPELINE / UNET HELPERS
#  KEY FIX: local_files_only=True  → never tries to reach huggingface.co
# ─────────────────────────────────────────────────────────────────────────────

def load_base_pipeline():
    """
    Load from the local HF cache only.
    Run download_model.py on a login node first if the cache is empty.
    """
    try:
        pipe = DDPMPipeline.from_pretrained(
            BASE_MODEL_ID,
            local_files_only=True,   # ← THE FIX
        )
    except Exception as e:
        raise RuntimeError(
            f"\n[ERROR] Could not load '{BASE_MODEL_ID}' from local cache.\n"
            f"  Cache dir : {os.environ.get('HF_HOME')}\n"
            f"  Fix       : run  python download_model.py  on a login node first.\n"
            f"  Original error: {e}"
        ) from e
    return pipe


def load_unet_fp32(pipe):
    model = UNet2DModel.from_config(pipe.unet.config)
    model.load_state_dict(pipe.unet.state_dict())
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_unet_from_checkpoint(ckpt_path: str, unet_config) -> nn.Module:
    """
    Load a fine-tuned UNet from a plain .pt checkpoint.

    Handles three common save formats:
      • full state-dict          { "weight_name": tensor, … }
      • wrapped state-dict       { "model": <state_dict> }
      • wrapped state-dict       { "model_state_dict": <state_dict> }
    """
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"\n[ERROR] Finetuned checkpoint not found: {ckpt_path}\n"
            f"  Set FINETUNED_CKPT env-var to the correct path."
        )
    print(f"  Loading finetuned checkpoint: {ckpt_path}")
    raw = torch.load(ckpt_path, map_location="cpu")

    # Unwrap common wrapper keys
    if isinstance(raw, dict):
        for key in ("model", "model_state_dict", "state_dict", "unet"):
            if key in raw and isinstance(raw[key], dict):
                raw = raw[key]
                print(f"  → unwrapped key '{key}'")
                break

    model = UNet2DModel.from_config(unet_config)
    missing, unexpected = model.load_state_dict(raw, strict=False)
    if missing:
        print(f"  [WARN] missing keys  ({len(missing)}): {missing[:5]} …")
    if unexpected:
        print(f"  [WARN] unexpected keys ({len(unexpected)}): {unexpected[:5]} …")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def get_cifar_loader(batch_size=EVAL_BATCH):
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,)*3, (0.5,)*3),
    ])
    ds = CIFAR10(root=CIFAR_ROOT, train=True, download=True, transform=tfm)
    workers = int(os.environ.get("DATALOADER_WORKERS", "4"))
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=workers, pin_memory=True, drop_last=False)


# ─────────────────────────────────────────────────────────────────────────────
#  SAMPLERS  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def ddim_sample(model, sched_cfg, num_steps, batch_size, device, seed=None):
    scheduler = DDIMScheduler.from_config(sched_cfg)
    scheduler.set_timesteps(num_steps)
    timesteps = scheduler.timesteps.to(device)
    alphas    = scheduler.alphas_cumprod.to(device)

    gen = torch.Generator(device=device).manual_seed(seed) if seed is not None else None
    x   = (torch.randn(batch_size, 3, 32, 32, device=device, generator=gen)
           if gen else torch.randn(batch_size, 3, 32, 32, device=device))

    for i, t in enumerate(timesteps):
        t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
        with torch.no_grad():
            x_in       = x.to(next(model.parameters()).dtype)
            noise_pred = model(x_in, t_batch).sample.float()

        a_s = alphas[t].view(-1, 1, 1, 1)
        if i == len(timesteps) - 1:
            x = (x - torch.sqrt(1 - a_s) * noise_pred) / torch.sqrt(a_s)
            break
        next_t = timesteps[i + 1]
        a_e    = alphas[next_t].view(-1, 1, 1, 1)
        x0_p   = (x - torch.sqrt(1 - a_s) * noise_pred) / torch.sqrt(a_s)
        x      = torch.sqrt(a_e) * x0_p + torch.sqrt(1 - a_e) * noise_pred

    return x.clamp(-1, 1)


# ─────────────────────────────────────────────────────────────────────────────
#  QUANTIZATION RECIPES  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def quant_fp32(model):
    return model.cpu()

def quant_fp16(model):
    return model.cpu().half()

def quant_bf16(model):
    return model.cpu().to(torch.bfloat16)

def quant_int4_bnb(model):
    if not HAS_BNB:
        raise RuntimeError("bitsandbytes not installed – cannot do INT4.")

    def _replace_linear(module):
        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                new = bnb.nn.Linear4bit(
                    child.in_features, child.out_features,
                    bias=child.bias is not None,
                    quant_type="nf4",
                    compute_dtype=torch.float16,
                )
                new.weight = bnb.nn.Params4bit(
                    child.weight.data, requires_grad=False, quant_type="nf4"
                )
                if child.bias is not None:
                    new.bias = nn.Parameter(child.bias.data)
                setattr(module, name, new)
            else:
                _replace_linear(child)

    q = copy.deepcopy(model)
    _replace_linear(q)
    return q

def quant_int4_emulated(model):
    q = copy.deepcopy(model).cpu()
    with torch.no_grad():
        for name, param in q.named_parameters():
            if param.ndim < 2:
                continue
            max_val = param.abs().amax(
                dim=tuple(range(1, param.ndim)), keepdim=True
            ).clamp(min=1e-8)
            scale = max_val / 7.0
            q_int = (param / scale).round().clamp(-8, 7)
            param.copy_(q_int * scale)
    return q


# ─────────────────────────────────────────────────────────────────────────────
#  MODEL SIZE HELPERS  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def model_size_mb(model):
    total = 0
    for p in model.parameters():
        try:
            total += p.nelement() * p.element_size()
        except Exception:
            total += p.nelement() * 1
    return total / (1024 ** 2)

def count_params(model):
    return sum(p.numel() for p in model.parameters())


# ─────────────────────────────────────────────────────────────────────────────
#  CORE: EVALUATE QUALITY  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(name, model, sched_cfg, real_loader, num_samples=NUM_EVAL, steps=DDIM_STEPS):
    print(f"\n{'═'*62}")
    print(f"  Evaluating quality: {name}")
    print(f"{'═'*62}")

    eval_device = DEVICE
    model = model.to(eval_device)
    model.eval()

    fid_m = FrechetInceptionDistance(feature=2048, normalize=True).to(DEVICE)
    is_m  = InceptionScore(normalize=True).to(DEVICE)
    # pr_m  = PrecisionRecall(k=3, device=DEVICE)

    real_list, n = [], 0
    for imgs, _ in tqdm(real_loader, desc="  Real → FID"):
        if n >= num_samples:
            break
        imgs_01 = (imgs.to(DEVICE) + 1.0) / 2.0
        fid_m.update(imgs_01, real=True)
        real_list.append(imgs_01.cpu())
        n += imgs.shape[0]
    real_t = torch.cat(real_list)[:num_samples]

    fake_list, n = [], 0
    while n < num_samples:
        bs = min(GEN_BATCH, num_samples - n)
        with torch.no_grad():
            s = ddim_sample(model, sched_cfg, steps, bs, eval_device)
        s01 = (s.float() + 1.0) / 2.0
        fid_m.update(s01.to(DEVICE), real=False)
        is_m.update(s01.to(DEVICE))
        fake_list.append(s01.cpu())
        n += bs
        print(f"  Generated {n}/{num_samples}", end="\r")
    fake_t = torch.cat(fake_list)[:num_samples]

    fid_val      = fid_m.compute().item()
    is_mu, is_sig = is_m.compute()
    # prec, rec    = pr_m.compute(real_t, fake_t)

    res = dict(name=name,
               steps=steps,
               FID=round(fid_val, 4),
               IS_mean=round(is_mu.item(), 4),
               IS_std=round(is_sig.item(), 4))
            #    Precision=round(prec, 4),
            #    Recall=round(rec, 4))

    # print(f"\n  FID={res['FID']:.4f}  "f"IS={res['IS_mean']:.4f}±{res['IS_std']:.4f}  "f"P={res['Precision']:.4f}  R={res['Recall']:.4f}")
    print(f"\n  FID={res['FID']:.4f}  "f"IS={res['IS_mean']:.4f}±{res['IS_std']:.4f}  ")

    del fid_m, is_m
    model.cpu()
    torch.cuda.empty_cache()
    return res


# ─────────────────────────────────────────────────────────────────────────────
#  CORE: TIMING BENCHMARK  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_timing(name, model, sched_cfg, save_tag, steps=DDIM_STEPS):
    print(f"\n  ⏱  Timing: {name}")
    eval_device = DEVICE
    model = model.to(eval_device)
    model.eval()
    timing = {}

    for bs in TIMING_BATCHES:
        times = []
        for rep in range(TIMING_REPS):
            if eval_device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                imgs = ddim_sample(model, sched_cfg, steps, bs, eval_device)
            if eval_device == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

            if rep == 0 and bs in SAVE_BATCHES:
                imgs_01 = (imgs.float() + 1.0) / 2.0
                tag = save_tag.replace(" ", "_")
                vutils.save_image(
                    imgs_01,
                    os.path.join(OUTPUT_DIR, "images", f"{tag}_bs{bs}.png"),
                    nrow=min(bs, 4), padding=2,
                )

        avg = float(np.mean(times))
        timing[bs] = {"total_s": round(avg, 4), "per_image_s": round(avg / bs, 6)}
        print(f"    BS={bs:3d}  {avg:.3f}s  ({avg/bs*1000:.2f} ms/img)")

    model.cpu()
    torch.cuda.empty_cache()
    return timing


# ─────────────────────────────────────────────────────────────────────────────
#  QUANTIZATION CONFIG TABLE
#  Each entry: (display_name, model, save_tag, notes, ddim_steps)
# ─────────────────────────────────────────────────────────────────────────────

def _quant_variants(base_model, prefix, save_prefix, steps):
    """
    Build all quantisation variants for one base model.
    Returns a list of 5-tuples:
        (display_name, quantised_model, save_tag, notes, steps)
    """
    configs = []

    configs.append((f"{prefix} FP32",
                    quant_fp32(copy.deepcopy(base_model)),
                    f"{save_prefix}_fp32",
                    "No quantization. Full 32-bit weights and activations.",
                    steps))

    configs.append((f"{prefix} FP16",
                    quant_fp16(copy.deepcopy(base_model)),
                    f"{save_prefix}_fp16",
                    "Half-precision float. ~2× memory reduction vs FP32.",
                    steps))

    if DEVICE != "cuda" or torch.cuda.is_bf16_supported():
        configs.append((f"{prefix} BF16",
                        quant_bf16(copy.deepcopy(base_model)),
                        f"{save_prefix}_bf16",
                        "BFloat16. Same memory as FP16 but wider dynamic range.",
                        steps))

    if HAS_BNB and DEVICE == "cuda":
        try:
            configs.append((f"{prefix} INT4-NF4",
                            quant_int4_bnb(copy.deepcopy(base_model)),
                            f"{save_prefix}_int4_bnb",
                            "NF4 weight-only quantization via bitsandbytes. CUDA only.",
                            steps))
        except Exception as e:
            print(f"  [WARN] bitsandbytes INT4 failed: {e}")

    configs.append((f"{prefix} INT4-Emu",
                    quant_int4_emulated(copy.deepcopy(base_model)),
                    f"{save_prefix}_int4_emu",
                    "Simulated INT4 (weights rounded to 4-bit, stored as FP32).",
                    steps))

    return configs


def build_quant_configs(base_model):
    """Base google/ddpm-cifar10-32 at DDIM_STEPS=50."""
    print("\n  Building base-model (DDIM-50) variants …")
    return _quant_variants(base_model,
                           prefix="Base", save_prefix="base",
                           steps=DDIM_STEPS)


def build_finetuned_configs(unet_config):
    """Finetuned checkpoint at FINETUNED_STEPS=8."""
    print(f"\n  Building finetuned-model (DDIM-{FINETUNED_STEPS}) variants …")
    ft_base = load_unet_from_checkpoint(FINETUNED_CKPT, unet_config)
    return _quant_variants(ft_base,
                           prefix="FT-8s", save_prefix="ft8s",
                           steps=FINETUNED_STEPS)


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("Loading base pipeline …")
    pipe      = load_base_pipeline()
    sched_cfg = pipe.scheduler.config
    base      = load_unet_fp32(pipe)

    real_loader  = get_cifar_loader(batch_size=EVAL_BATCH)

    # ── Build all configs (base @ 50 steps  +  finetuned @ 8 steps) ──────────
    base_configs = build_quant_configs(base)
    ft_configs   = build_finetuned_configs(pipe.unet.config)
    all_configs  = base_configs + ft_configs

    # ── Model-size table ──────────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"  {'Model':<30} {'Steps':>5}  {'Params':>10}  {'Size (MB)':>10}")
    print(f"{'─'*70}")
    for name, model, _, notes, steps in all_configs:
        print(f"  {name:<30} {steps:>5}  {count_params(model):>10,}  "
              f"{model_size_mb(model):>8.1f} MB")
    print(f"{'─'*70}\n")

    all_results = []
    all_timing  = {}

    for name, model, save_tag, _, steps in all_configs:
        res = evaluate_model(name, model, sched_cfg, real_loader,
                             num_samples=NUM_EVAL, steps=steps)
        all_results.append(res)
        all_timing[name] = benchmark_timing(name, model, sched_cfg,
                                            save_tag=save_tag, steps=steps)

    # ─────────────────────────────────────────────────────────────────────────
    #  SUMMARY TABLES  – printed per model family so ΔFID is meaningful
    # ─────────────────────────────────────────────────────────────────────────
    def _quality_table(results, title, fp32_key_substr):
        fp32_r = next((r for r in results if fp32_key_substr in r["name"]), results[0])
        cw = 32
        print("\n\n" + "═"*106)
        print(f"  {title}")
        print("═"*106)
        print(f"  {'Model':<{cw}}  {'Steps':>5}  {'FID':>8}  {'ΔFID':>8}  "
              f"{'IS':>10}  {'Prec':>8}  {'Recall':>8}")
        print("─"*106)
        for r in results:
            delta = r["FID"] - fp32_r["FID"]
            ds    = f"{delta:+.4f}" if delta != 0 else "  0.0000"
            print(f"  {r['name']:<{cw}}  {r['steps']:>5}  {r['FID']:>8.4f}  {ds:>8}  "
                  f"{r['IS_mean']:>6.4f}±{r['IS_std']:<4.4f}  "
                  f"{r['Precision']:>8.4f}  {r['Recall']:>8.4f}")

    def _speed_table(results, timing, title, fp32_key_substr):
        cw     = 32
        bs_hdr = "   ".join([f"BS={bs:>2}" for bs in TIMING_BATCHES])
        fp32_name = next(r["name"] for r in results if fp32_key_substr in r["name"])
        print("\n\n" + "═"*106)
        print(f"  {title}  –  ms/image  (speedup vs {fp32_name})")
        print("═"*106)
        print(f"  {'Model':<{cw}}  {bs_hdr}")
        print("─"*106)
        for r in results:
            n = r["name"]
            row_parts = []
            for bs in TIMING_BATCHES:
                ms     = timing[n][bs]["per_image_s"] * 1000
                ms_fp  = timing[fp32_name][bs]["per_image_s"] * 1000
                speedup = ms_fp / ms if ms > 0 else 0
                row_parts.append(f"{ms:>6.2f}ms({speedup:.2f}×)")
            print(f"  {n:<{cw}}  {'  '.join(row_parts)}")

    base_results = [r for r in all_results if r["name"].startswith("Base")]
    ft_results   = [r for r in all_results if r["name"].startswith("FT-8s")]

    _quality_table(base_results,
                   f"BASE MODEL — QUALITY  (DDIM-{DDIM_STEPS}, 10 k images vs CIFAR-10)",
                   "Base FP32")
    _quality_table(ft_results,
                   f"FINETUNED MODEL — QUALITY  (DDIM-{FINETUNED_STEPS}, 10 k images vs CIFAR-10)",
                   "FT-8s FP32")

    _speed_table(base_results, all_timing,
                 f"BASE MODEL — INFERENCE SPEED  (DDIM-{DDIM_STEPS})",
                 "Base FP32")
    _speed_table(ft_results, all_timing,
                 f"FINETUNED MODEL — INFERENCE SPEED  (DDIM-{FINETUNED_STEPS})",
                 "FT-8s FP32")

    # ── JSON report ───────────────────────────────────────────────────────────
    model_sizes = {
        n: {"size_mb": round(model_size_mb(m), 2),
            "params": count_params(m), "notes": notes, "ddim_steps": steps}
        for n, m, _, notes, steps in all_configs
    }
    report = dict(
        base_ddim_steps=DDIM_STEPS,
        finetuned_ddim_steps=FINETUNED_STEPS,
        finetuned_ckpt=FINETUNED_CKPT,
        num_eval_images=NUM_EVAL,
        device=DEVICE,
        quality_metrics=all_results,
        timing=all_timing,
        model_sizes=model_sizes,
    )
    path = os.path.join(OUTPUT_DIR, "quantization_report.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n✓  Report saved → {path}")
    print(f"✓  Images saved → {os.path.join(OUTPUT_DIR, 'images')}/")


if __name__ == "__main__":
    main()
