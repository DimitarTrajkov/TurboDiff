"""
33_benchmark_inference.py

Inference-time benchmark across models and batch sizes — a more precise
replacement for script 28.

Why this is more precise than script 28:
  1. Warm-up is redone for every (model, batch, steps) configuration, not once.
     cuDNN autotuning, lazy CUDA init, allocator growth and GPU clock ramp-up
     all distort the first iterations of each new input shape.
  2. Reports median + p90/p99, not just mean/std. GPU latency is right-skewed by
     OS/scheduler jitter, so the median is the stable summary.
  3. The starting noise is allocated ONCE, outside the timed region. Only the
     denoising loop (the model-bound work) is timed.
  4. Reports throughput (images/sec) and ms/image — the right metric for a batch
     sweep — alongside per-batch latency.
  5. cudnn.benchmark=True so kernels are autotuned for the fixed 32x32 shapes;
     the warm-up absorbs the one-off autotuning cost.
  6. torch.cuda.synchronize() brackets every timed iteration, and timing uses
     CUDA events (GPU-side), falling back to perf_counter on CPU.
  7. Peak GPU memory is reported per configuration.

Hardware notes (2x RTX 5000 Ada):
  - A single benchmark process should pin ONE GPU for clean numbers. Use
    --device cuda:0 / cuda:1 to benchmark each card; running on both at once in
    one process mixes streams and muddies the measurement.
  - To measure aggregate two-GPU throughput, launch two processes, e.g.
        CUDA_VISIBLE_DEVICES=0 python 33_benchmark_inference.py &
        CUDA_VISIBLE_DEVICES=1 python 33_benchmark_inference.py &
  - For maximum reproducibility, run on an idle GPU with locked clocks:
        sudo nvidia-smi -pm 1
        sudo nvidia-smi -i 0 -lgc <clock>      # lock graphics clock
    and keep the card cool to avoid thermal throttling mid-run.

Usage:
    python 33_benchmark_inference.py
    python 33_benchmark_inference.py --device cuda:1 --dtype fp16 --csv out.csv
    python 33_benchmark_inference.py --batch-sizes 1 16 64 256 --timed 100
"""

import argparse
import contextlib
import functools
import time

import numpy as np
import torch

from ddpm_arch import UNet2DModel, linear_alphas_cumprod, ddim_step


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


# Each entry: (label, weights, native_inference_steps, sampler).
#   weights : a checkpoint path, or a callable returning one (used to lazily fetch
#             the base model from the HF hub only when it is actually benchmarked).
#   sampler : "ddpm" = the original stochastic ancestral process (no DDIM shortcut);
#             "ddim" = deterministic few-step sampler.
# All entries share one UNet, so per-forward cost is identical; total latency
# scales with steps x batch.
MODELS = [
    ("base-ddpm-1000", fetch_base_weights, 1000, "ddpm"),
    ("base-ddim-25",   fetch_base_weights,   25, "ddim"),
    ("student-25", "./checkpoints/fast_professor_21_final.pt", 25, "ddim"),
    ("student-12", "./checkpoints/fast_professor_12step.pt",   12, "ddim"),
    ("student-8",  "./checkpoints/fast_professor_8step.pt",     8, "ddim"),
]

DEFAULT_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256]

DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}

NUM_TRAIN_TIMESTEPS = 1000


def load_state_dict_file(path, device):
    """Load a state dict from either a .safetensors or a .bin/.pt checkpoint."""
    if str(path).endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(path, device=str(device))
    return torch.load(path, map_location=device, weights_only=True)


def load_model(path, device):
    """Build the UNet and load a checkpoint. AttentionBlock accepts both diffusers key namings."""
    model = UNet2DModel().to(device)
    state = load_state_dict_file(path, device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def precompute_steps(alphas, n_steps, batch, device):
    """
    Pre-build everything the denoising loop needs, so the timed region contains
    no host<->device syncs (e.g. no .item() calls) — only model forwards + math.
    """
    timesteps = torch.linspace(NUM_TRAIN_TIMESTEPS - 1, 0, n_steps, dtype=torch.long)
    step_ints = [int(t) for t in timesteps]
    t_batches, a_s_list, a_e_list = [], [], []
    for i, ti in enumerate(step_ints):
        t_batches.append(torch.full((batch,), ti, device=device, dtype=torch.long))
        a_s_list.append(alphas[ti].view(1, 1, 1, 1))
        a_e_list.append(None if i == len(step_ints) - 1
                        else alphas[step_ints[i + 1]].view(1, 1, 1, 1))
    return t_batches, a_s_list, a_e_list


@torch.inference_mode()
def denoise(model, x0, t_batches, a_s_list, a_e_list, autocast):
    """Full multi-step DDIM denoise. x0 is never mutated in place, so it can be reused.

    inference_mode() disables autograd: without it PyTorch retains activations for
    the whole multi-step rollout, which both inflates memory (premature OOM) and is
    meaningless for an inference-time benchmark.
    """
    x = x0
    with autocast:
        for i in range(len(t_batches)):
            eps = model(x, t_batches[i])
            a_s = a_s_list[i]
            if a_e_list[i] is None:                      # final step -> predict x0
                x = (x - (1 - a_s).sqrt() * eps) / a_s.sqrt()
            else:
                x = ddim_step(eps, x, a_s, a_e_list[i])
    return x


def precompute_ddpm(alphas, batch, device, num_steps=NUM_TRAIN_TIMESTEPS):
    """
    Pre-build the DDPM ancestral-sampling constants for every integer timestep
    (num_steps-1 .. 0), so the timed loop has no host<->device syncs. Mirrors a
    linear-beta DDPM with variance_type=fixed_large, clip_sample=True.
    """
    one = torch.tensor(1.0, device=device)
    t_batches, consts = [], []
    for t in range(num_steps - 1, -1, -1):
        a_t = alphas[t]
        a_prev = alphas[t - 1] if t > 0 else one
        beta_t = 1 - a_t / a_prev
        coef1 = a_prev.sqrt() * beta_t / (1 - a_t)
        coef2 = (a_t / a_prev).sqrt() * (1 - a_prev) / (1 - a_t)
        t_batches.append(torch.full((batch,), t, device=device, dtype=torch.long))
        consts.append((a_t.view(1, 1, 1, 1), beta_t.view(1, 1, 1, 1),
                       coef1.view(1, 1, 1, 1), coef2.view(1, 1, 1, 1), t > 0))
    return t_batches, consts


@torch.inference_mode()
def denoise_ddpm(model, x0, t_batches, consts, autocast):
    """Original stochastic DDPM reverse process (no DDIM shortcut) — fresh noise per step."""
    x = x0
    with autocast:
        for i in range(len(t_batches)):
            eps = model(x, t_batches[i])
            a_t, beta_t, coef1, coef2, add_noise = consts[i]
            x0_pred = ((x - (1 - a_t).sqrt() * eps) / a_t.sqrt()).clamp(-1, 1)
            x = coef1 * x0_pred + coef2 * x
            if add_noise:
                x = x + beta_t.sqrt() * torch.randn_like(x)
    return x


def time_config(model, alphas, n_steps, batch, device, dtype,
                warmup, timed, sampler):
    """Return a dict of latency stats (ms) for one (model, steps, batch) config."""
    use_cuda = device.type == "cuda"
    autocast = (torch.autocast(device_type="cuda", dtype=dtype)
                if (use_cuda and dtype != torch.float32) else contextlib.nullcontext())

    if use_cuda:
        torch.cuda.reset_peak_memory_stats(device)

    noise = torch.randn(batch, 3, 32, 32, device=device)     # allocated once, outside timer

    # Build the sampler-specific per-step constants, then a zero-arg `run` closure.
    if sampler == "ddpm":
        t_batches, consts = precompute_ddpm(alphas, batch, device, n_steps)
        def run():
            return denoise_ddpm(model, noise, t_batches, consts, autocast)
    else:
        t_batches, a_s_list, a_e_list = precompute_steps(alphas, n_steps, batch, device)
        def run():
            return denoise(model, noise, t_batches, a_s_list, a_e_list, autocast)

    # A many-step generation already absorbs warm-up costs and has tiny run-to-run
    # variance, so cap repeats to keep total forwards (and wall time) bounded.
    if n_steps > 100:
        warmup, timed = min(warmup, 2), min(timed, 5)

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
        "p99_ms": float(np.percentile(times_ms, 99)),
        "mean_ms": float(times_ms.mean()),
        "std_ms": float(times_ms.std()),
        "ms_per_img": p50 / batch,
        "img_per_s": batch / (p50 / 1000.0),
        "peak_mb": peak_mb,
    }


def benchmark_model(label, weights, n_steps, sampler, alphas, device, dtype, args):
    """Benchmark one model across all requested batch sizes; print and return rows."""
    try:
        path = weights() if callable(weights) else weights   # lazily fetch base from HF
        model = load_model(path, device)
    except OSError as e:                                      # FileNotFoundError subclasses OSError
        print(f"{label:<15}  (skipped — weights unavailable: {e})")
        return []

    rows = []
    for batch in args.batch_sizes:
        try:
            stats = time_config(model, alphas, n_steps, batch, device, dtype,
                                args.warmup, args.timed, sampler)
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            print(f"{label:<15}{n_steps:>6}{batch:>7}  (OOM — skipped)")
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue

        print(f"{label:<15}{n_steps:>6}{batch:>7}{stats['p50_ms']:>10.2f}"
              f"{stats['p90_ms']:>10.2f}{stats['ms_per_img']:>9.3f}"
              f"{stats['img_per_s']:>10.1f}{stats['peak_mb']:>10.1f}")
        rows.append({"model": label, "steps": n_steps, "batch": batch, **stats})

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows


def write_csv(path, rows):
    import csv
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {path}")


def configure_device(device, args):
    """Apply backend flags and print a header line for the chosen device."""
    if device.type == "cuda":
        # Pin the active device so CUDA events, streams and kernels all live on it
        # (otherwise Event.record() defaults to cuda:0 and elapsed_time() fails).
        torch.cuda.set_device(device)
        # Autotune kernels for the fixed input shapes; allow TF32 on Ada (fp32 path).
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        name = torch.cuda.get_device_name(device)
        print(f"Device: {device} ({name}) | dtype: {args.dtype} | "
              f"warmup={args.warmup} timed={args.timed}")
    else:
        print(f"Device: {device} (CPU — CUDA-event timing unavailable, using perf_counter) | "
              f"dtype: {args.dtype}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="fp32", choices=list(DTYPES))
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=DEFAULT_BATCH_SIZES)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--timed", type=int, default=50)
    parser.add_argument("--csv", default=None, help="optional path to write a CSV of results")
    parser.add_argument("--models", nargs="+", default=None,
                        help="only run models whose label contains one of these substrings "
                             "(e.g. --models student  or  --models base-ddim)")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = DTYPES[args.dtype]
    configure_device(device, args)

    alphas = linear_alphas_cumprod().to(device)

    header = (f"{'model':<15}{'steps':>6}{'batch':>7}{'p50 ms':>10}{'p90 ms':>10}"
              f"{'ms/img':>9}{'img/s':>10}{'peak MB':>10}")
    print("\n" + header)
    print("-" * len(header))

    rows = []
    for label, weights, n_steps, sampler in MODELS:
        if args.models and not any(m in label for m in args.models):
            continue
        rows.extend(benchmark_model(label, weights, n_steps, sampler, alphas, device, dtype, args))

    if args.csv and rows:
        write_csv(args.csv, rows)


if __name__ == "__main__":
    main()
