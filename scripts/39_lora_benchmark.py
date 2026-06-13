"""
39_lora_benchmark.py

Benchmark the per-step LoRA students trained by script 37:
  - Inference time  — the precise methodology of script 33 (CUDA-event timing,
    per-(model,batch) warm-up, median + p90, ms/img, img/s, peak memory).
  - FID and IS      — the same torchmetrics path as benchmark.py / metrics_utils
    (FrechetInceptionDistance(feature=2048, normalize=True), InceptionScore).
    Precision/Recall are reported too, computed exactly as in Appendix A.4
    (Inception-2048 features, k=3) so the numbers line up with that table.

The LoRA checkpoints are not plain state dicts — they carry a bank of per-step
heads and require head switching during sampling — so this script drives them
through script 37's own loader (`load_lora_student`) and sampler (`generate_lora`),
which it imports directly (37 is digit-prefixed, hence importlib).

Usage:
    # timing only (no CIFAR needed), all three best LoRA students:
    python 39_lora_benchmark.py --device cuda:2 \
        --checkpoints checkpoints_v2/lora_student_25step.pt \
                      checkpoints_v2/lora_student_12step.pt \
                      checkpoints_v3/lora_student_8step.pt

    # timing + FID/IS/P&R on 10k samples:
    python 39_lora_benchmark.py --device cuda:2 --eval-samples 10000 \
        --checkpoints checkpoints_v3/lora_student_8step.pt

    # quick metric check:
    python 39_lora_benchmark.py --device cuda:2 --eval-samples 2000 --batch-sizes 1 32 \
        --checkpoints checkpoints_v3/lora_student_8step.pt
"""

import argparse
import importlib.util
import os
import time

import numpy as np
import torch

from ddpm_arch import linear_alphas_cumprod, ddim_step

DEFAULT_BATCH_SIZES = [1, 4, 8, 16, 32, 64]


def _import_script37():
    """Import the LoRA machinery from 37_lora_progressive_distillation.py (digit-prefixed)."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "37_lora_progressive_distillation.py")
    spec = importlib.util.spec_from_file_location("lora37", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


L = _import_script37()


# ─────────────────────────────────────────────────────────────
# Inference-time benchmark (script-33 methodology, per-step heads)
# ─────────────────────────────────────────────────────────────
@torch.inference_mode()
def time_config(model, schedule, alphas, batch, device, warmup, timed):
    """Latency stats (ms) for one batch size, activating head i at step i."""
    use_cuda = device.type == "cuda"
    if use_cuda:
        torch.cuda.reset_peak_memory_stats(device)

    n = len(schedule)
    # Precompute per-step constants so the timed loop has no host<->device syncs.
    steps = []
    for i in range(n):
        t = int(schedule[i])
        a_s = alphas[t].view(1, 1, 1, 1)
        a_e = None if i == n - 1 else alphas[int(schedule[i + 1])].view(1, 1, 1, 1)
        t_batch = torch.full((batch,), t, device=device, dtype=torch.long)
        steps.append((i, t_batch, a_s, a_e))

    noise = torch.randn(batch, 3, 32, 32, device=device)     # allocated once, outside timer

    def run():
        x = noise
        for i, t_batch, a_s, a_e in steps:
            L.set_active_head(model, i)                       # genuine per-step LoRA cost
            eps = model(x, t_batch)
            if a_e is None:
                x = (x - (1 - a_s).sqrt() * eps) / a_s.sqrt()
            else:
                x = ddim_step(eps, x, a_s, a_e)
        return x

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


def benchmark_timing(label, model, schedule, alphas, device, args):
    n_steps = len(schedule)
    rows = []
    for batch in args.batch_sizes:
        try:
            s = time_config(model, schedule, alphas, batch, device, args.warmup, args.timed)
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            print(f"{label:<22}{n_steps:>6}{batch:>7}  (OOM — skipped)")
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue
        print(f"{label:<22}{n_steps:>6}{batch:>7}{s['p50_ms']:>10.2f}{s['p90_ms']:>10.2f}"
              f"{s['ms_per_img']:>9.3f}{s['img_per_s']:>10.1f}{s['peak_mb']:>10.1f}")
        rows.append({"model": label, "steps": n_steps, "batch": batch, **s})
    return rows


# ─────────────────────────────────────────────────────────────
# FID / IS / Precision / Recall
# ─────────────────────────────────────────────────────────────
@torch.inference_mode()
def generate_fakes(model, schedule, alphas, count, batch, device):
    """Generate `count` LoRA samples; return uint8 (count,3,32,32) on CPU."""
    imgs, done = [], 0
    while done < count:
        bs = min(batch, count - done)
        fake01 = (L.generate_lora(model, schedule, alphas, bs, device) + 1.0) / 2.0
        imgs.append((fake01 * 255).round().byte().cpu())
        done += bs
        print(f"  generated {done}/{count}", end="\r")
    print()
    return torch.cat(imgs, dim=0)


def evaluate_quality(label, model, schedule, alphas, real_imgs01, real_feats,
                     extractor, args, device):
    from metrics_utils import compute_metrics
    fakes = generate_fakes(model, schedule, alphas, args.eval_samples, args.eval_batch, device)
    m = compute_metrics(fakes, real_imgs01, real_feats, extractor, device,
                        args.eval_batch, args.knn)
    print(f"{label}: FID {m['fid']:.3f} | IS {m['is_mean']:.3f}±{m['is_std']:.3f} | "
          f"Precision {m['precision']:.3f} | Recall {m['recall']:.3f}")
    return m


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoints", nargs="+", required=True,
                        help="one or more LoRA student checkpoints from script 37")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=DEFAULT_BATCH_SIZES)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--timed", type=int, default=50)
    parser.add_argument("--eval-samples", type=int, default=0,
                        help="if >0, also compute FID/IS/Precision/Recall on this many samples")
    parser.add_argument("--eval-batch", type=int, default=250)
    parser.add_argument("--knn", type=int, default=3)
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f"Device: {device} ({torch.cuda.get_device_name(device)}) | "
              f"warmup={args.warmup} timed={args.timed}")
    else:
        print(f"Device: {device} (CPU — perf_counter timing) | warmup={args.warmup} timed={args.timed}")

    alphas = linear_alphas_cumprod().to(device)

    # Real reference for FID/IS/P&R, prepared once and reused (only if evaluating).
    real_imgs01 = real_feats = extractor = None
    if args.eval_samples > 0:
        from metrics_utils import prepare_real
        print(f"\nLoading {args.eval_samples} real images and extracting features…")
        extractor, real_imgs01, real_feats = prepare_real(args.eval_samples, args.data_root, device)

    header = (f"{'model':<22}{'steps':>6}{'batch':>7}{'p50 ms':>10}{'p90 ms':>10}"
              f"{'ms/img':>9}{'img/s':>10}{'peak MB':>10}")
    print("\n--- Inference time ---\n" + header + "\n" + "-" * len(header))

    quality = []
    for ckpt in args.checkpoints:
        if not os.path.exists(ckpt):
            print(f"{os.path.basename(ckpt):<22}  (skipped — file not found: {ckpt})")
            continue
        model, schedule = L.load_lora_student(ckpt, device)
        model.eval()
        label = os.path.basename(ckpt).replace(".pt", "")

        benchmark_timing(label, model, schedule, alphas, device, args)

        if args.eval_samples > 0:
            torch.manual_seed(args.seed)
            quality.append((label, evaluate_quality(label, model, schedule, alphas,
                                                    real_imgs01, real_feats, extractor, args, device)))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if quality:
        print("\n--- Quality summary ---")
        print(f"{'model':<28}{'FID':>9}{'IS':>16}{'Precision':>11}{'Recall':>9}")
        print("-" * 73)
        for label, m in quality:
            print(f"{label:<28}{m['fid']:>9.3f}{m['is_mean']:>10.3f}±{m['is_std']:<5.3f}"
                  f"{m['precision']:>11.3f}{m['recall']:>9.3f}")


if __name__ == "__main__":
    main()
