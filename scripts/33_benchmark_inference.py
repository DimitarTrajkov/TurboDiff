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
import time

import numpy as np
import torch

from ddpm_arch import UNet2DModel, linear_alphas_cumprod, ddim_step, WEIGHTS_PATH


# Each entry: (label, checkpoint_path, native_inference_steps).
# All share the same UNet architecture, so per-forward cost is identical; total
# latency scales with steps x batch. Missing checkpoints are skipped with a note.
MODELS = [
    ("base-google", WEIGHTS_PATH,                 30),
    ("student-25",  "./checkpoints/fast_professor_21_final.pt", 25),
    ("student-12",  "./checkpoints/fast_professor_12step.pt",   12),
    ("student-8",   "./checkpoints/fast_professor_8step.pt",     8),
]

DEFAULT_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256]

DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}

NUM_TRAIN_TIMESTEPS = 1000


def load_model(path, device):
    """Build the UNet and load a checkpoint (state_dict keys match the diffusers model)."""
    model = UNet2DModel().to(device)
    state = torch.load(path, map_location=device, weights_only=True)
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


def denoise(model, x0, t_batches, a_s_list, a_e_list, autocast):
    """Full multi-step DDIM denoise. x0 is never mutated in place, so it can be reused."""
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


def time_config(model, alphas, n_steps, batch, device, dtype,
                warmup, timed):
    """Return a dict of latency stats (ms) for one (model, steps, batch) config."""
    use_cuda = device.type == "cuda"
    autocast = (torch.autocast(device_type="cuda", dtype=dtype)
                if (use_cuda and dtype != torch.float32) else contextlib.nullcontext())

    if use_cuda:
        torch.cuda.reset_peak_memory_stats(device)

    noise = torch.randn(batch, 3, 32, 32, device=device)     # allocated once, outside timer
    t_batches, a_s_list, a_e_list = precompute_steps(alphas, n_steps, batch, device)

    # Per-configuration warm-up (autotuning, allocation, clock ramp).
    for _ in range(warmup):
        denoise(model, noise, t_batches, a_s_list, a_e_list, autocast)
    if use_cuda:
        torch.cuda.synchronize(device)

    times_ms = []
    for _ in range(timed):
        if use_cuda:
            torch.cuda.synchronize(device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            denoise(model, noise, t_batches, a_s_list, a_e_list, autocast)
            end.record()
            torch.cuda.synchronize(device)
            times_ms.append(start.elapsed_time(end))
        else:
            t0 = time.perf_counter()
            denoise(model, noise, t_batches, a_s_list, a_e_list, autocast)
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


def benchmark_model(label, path, n_steps, alphas, device, dtype, args):
    """Benchmark one model across all requested batch sizes; print and return rows."""
    try:
        model = load_model(path, device)
    except FileNotFoundError:
        print(f"{label:<13}  (skipped — checkpoint not found: {path})")
        return []

    rows = []
    for batch in args.batch_sizes:
        try:
            stats = time_config(model, alphas, n_steps, batch, device, dtype,
                                args.warmup, args.timed)
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            print(f"{label:<13}{n_steps:>6}{batch:>7}  (OOM — skipped)")
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue

        print(f"{label:<13}{n_steps:>6}{batch:>7}{stats['p50_ms']:>10.2f}"
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
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = DTYPES[args.dtype]
    configure_device(device, args)

    alphas = linear_alphas_cumprod().to(device)

    header = (f"{'model':<13}{'steps':>6}{'batch':>7}{'p50 ms':>10}{'p90 ms':>10}"
              f"{'ms/img':>9}{'img/s':>10}{'peak MB':>10}")
    print("\n" + header)
    print("-" * len(header))

    rows = []
    for label, path, n_steps in MODELS:
        rows.extend(benchmark_model(label, path, n_steps, alphas, device, dtype, args))

    if args.csv and rows:
        write_csv(args.csv, rows)


if __name__ == "__main__":
    main()
