"""
37_lora_progressive_distillation.py

Progressive distillation with per-step LoRA heads instead of full fine-tuning.

Idea: keep the distillation logic of scripts 30/31/32 (the teacher traverses a
jump in several small DDIM steps, the student learns to cover it in one), but
FREEZE the base UNet and train a bank of LoRA adapters — one per inference step,
i.e. one per noise level. The N-step student therefore owns N LoRA heads, and at
inference step i, head i is activated before the model is evaluated.

Chain (mirrors the original): base(1000) -> 25 heads -> 12 heads -> 8 heads.
Each stage's teacher is the previous stage's model (base + its heads); the new
student is a fresh frozen base with N zero-initialised heads, warm-started from
the teacher's nearest head (the LoRA equivalent of "student = deepcopy(teacher)").

Adaptations vs. scripts 30/31/32 that per-step heads require:
  - The student trains on its EXACT inference schedule (linspace(999,0,N), the
    same grid ddpm_arch.generate_n_steps uses), not at random offsets of a finer
    grid, so head i always trains at the noise level where it is used.
  - One step index (= one head) is sampled PER BATCH, not per sample, so a single
    adapter is active for the whole forward pass.
  - The teacher traverses each student jump in K DDIM sub-steps; at every
    sub-step it activates its own head nearest to that timestep. Heads are
    noise-level experts — the model predicts eps and the jump size is DDIM math,
    so querying the teacher off its own schedule is well-defined.
  - Only LoRA parameters train (a few % of the model), so the LR is higher
    (1e-4) than the full-finetune scripts (1e-5).
  - The head at t=0 (the final x0-extraction eval) is never hit by a jump loss;
    it stays zero-initialised, which exactly reproduces the base model there.

Usage:
    python 37_lora_progressive_distillation.py --device cuda:2                # full 25->12->8 chain
    python 37_lora_progressive_distillation.py --device cuda:2 --eval-samples 10000   # + metrics per stage
    python 37_lora_progressive_distillation.py --stages 12 8 \
        --teacher ./checkpoints/lora_student_25step.pt --device cuda:2        # resume mid-chain
    python 37_lora_progressive_distillation.py --eval-only \
        --checkpoint ./checkpoints/lora_student_8step.pt --device cuda:2
    python 37_lora_progressive_distillation.py --rank 8 --targets attn temb conv   # more capacity
"""

import argparse
import functools
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from ddpm_arch import UNet2DModel, linear_alphas_cumprod, ddim_step

NUM_TRAIN_TIMESTEPS = 1000
ATTN_LINEAR_NAMES = {"query", "key", "value", "proj_attn"}
RESNET_CONV_NAMES = {"conv1", "conv2"}


# ─────────────────────────────────────────────────────────────
# Base model (weights from the HF hub, as in scripts 35/36)
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
# Multi-head LoRA wrappers
# ─────────────────────────────────────────────────────────────
class MultiLoRALinear(nn.Module):
    """nn.Linear with a bank of LoRA heads; exactly one head is active per forward.

    lora_B is zero-initialised, so a fresh head is an exact identity (= base model).
    Scaling is 1 (alpha = rank), so head magnitude is learned directly.
    """
    def __init__(self, base: nn.Linear, num_heads: int, rank: int):
        super().__init__()
        self.base = base
        self.rank = rank
        self.lora_A = nn.Parameter(
            torch.randn(num_heads, rank, base.in_features) / math.sqrt(base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(num_heads, base.out_features, rank))
        self.active = 0

    def forward(self, x):
        h = F.linear(F.linear(x, self.lora_A[self.active]), self.lora_B[self.active])
        return self.base(x) + h


class MultiLoRAConv2d(nn.Module):
    """nn.Conv2d with a bank of LoRA heads (down: kxk conv to rank, up: 1x1 conv)."""
    def __init__(self, base: nn.Conv2d, num_heads: int, rank: int):
        super().__init__()
        self.base = base
        self.rank = rank
        kh, kw = base.kernel_size
        fan_in = base.in_channels * kh * kw
        self.lora_A = nn.Parameter(
            torch.randn(num_heads, rank, base.in_channels, kh, kw) / math.sqrt(fan_in))
        self.lora_B = nn.Parameter(torch.zeros(num_heads, base.out_channels, rank, 1, 1))
        self.active = 0

    def forward(self, x):
        h = F.conv2d(x, self.lora_A[self.active],
                     stride=self.base.stride, padding=self.base.padding)
        h = F.conv2d(h, self.lora_B[self.active])
        return self.base(x) + h


def add_lora_heads(model, num_heads, rank, targets):
    """Wrap the target sub-layers with LoRA banks and freeze everything else.

    targets ⊆ {"attn", "temb", "conv"}:
      attn -> the query/key/value/proj_attn Linears of every AttentionBlock
      temb -> the time_emb_proj Linear of every ResnetBlock2D
      conv -> the conv1/conv2 3x3 convs of every ResnetBlock2D
    """
    for module in list(model.modules()):                     # snapshot: don't revisit new wrappers
        for name, child in list(module.named_children()):
            wrap = None
            if isinstance(child, nn.Linear):
                if ("attn" in targets and name in ATTN_LINEAR_NAMES) or \
                   ("temb" in targets and name == "time_emb_proj"):
                    wrap = MultiLoRALinear(child, num_heads, rank)
            elif isinstance(child, nn.Conv2d) and "conv" in targets and name in RESNET_CONV_NAMES:
                wrap = MultiLoRAConv2d(child, num_heads, rank)
            if wrap is not None:
                setattr(module, name, wrap.to(next(child.parameters()).device))
    for n, p in model.named_parameters():
        p.requires_grad = "lora_" in n
    return model


def set_active_head(model, i):
    for m in model.modules():
        if isinstance(m, (MultiLoRALinear, MultiLoRAConv2d)):
            m.active = i


def lora_state(model):
    return {k: v for k, v in model.state_dict().items() if "lora_" in k}


def load_lora_student(ckpt_path, device):
    """Rebuild base + heads from a checkpoint saved by this script."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model = load_base_model(device)
    add_lora_heads(model, ckpt["n_steps"], ckpt["rank"], set(ckpt["targets"]))
    result = model.load_state_dict(ckpt["lora_state"], strict=False)
    assert not result.unexpected_keys, f"unexpected keys: {result.unexpected_keys[:5]}"
    missing_lora = [k for k in result.missing_keys if "lora_" in k]
    assert not missing_lora, f"missing LoRA keys: {missing_lora[:5]}"
    return model, ckpt["schedule"].to(device)


def warm_start_from_teacher(student, s_sched, teacher, t_sched):
    """Copy each student head from the teacher head nearest in timestep."""
    t_mods = dict(teacher.named_modules())
    nearest = [int((t_sched - int(t)).abs().argmin()) for t in s_sched]
    copied = 0
    for name, sm in student.named_modules():
        if not isinstance(sm, (MultiLoRALinear, MultiLoRAConv2d)):
            continue
        tm = t_mods.get(name)
        if type(tm) is not type(sm) or tm.lora_A.shape[1:] != sm.lora_A.shape[1:]:
            continue
        with torch.no_grad():
            for i, j in enumerate(nearest):
                sm.lora_A[i].copy_(tm.lora_A[j])
                sm.lora_B[i].copy_(tm.lora_B[j])
        copied += 1
    print(f"  warm-started {copied} LoRA modules from nearest teacher heads")


# ─────────────────────────────────────────────────────────────
# Schedules, teacher queries, sampling
# ─────────────────────────────────────────────────────────────
def make_schedule(n_steps, device):
    """Identical grid to ddpm_arch.generate_n_steps, so heads align with the sampler."""
    return torch.linspace(NUM_TRAIN_TIMESTEPS - 1, 0, n_steps, dtype=torch.long, device=device)


def teacher_eps(teacher, t_sched, x, t):
    """Query the teacher at timestep t; if it has heads, use the one nearest to t."""
    if t_sched is not None:
        set_active_head(teacher, int((t_sched - t).abs().argmin()))
    t_batch = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
    return teacher(x, t_batch)


@torch.inference_mode()
def generate_lora(model, schedule, alphas, batch_size, device):
    """DDIM sampling along `schedule`, activating head i at step i."""
    x = torch.randn(batch_size, 3, 32, 32, device=device)
    n = len(schedule)
    for i in range(n):
        set_active_head(model, i)
        t = int(schedule[i])
        t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
        eps = model(x, t_batch)
        a_s = alphas[t].view(1, 1, 1, 1)
        if i == n - 1:                                       # final eval at t=0 -> direct x0
            x = (x - (1 - a_s).sqrt() * eps) / a_s.sqrt()
        else:
            a_e = alphas[int(schedule[i + 1])].view(1, 1, 1, 1)
            x = ddim_step(eps, x, a_s, a_e)
    return x.clamp(-1, 1)


# ─────────────────────────────────────────────────────────────
# One distillation stage
# ─────────────────────────────────────────────────────────────
def run_stage(n_steps, teacher, teacher_sched, substeps, loss_type, args, device, stage_seed):
    torch.manual_seed(stage_seed)
    alphas = linear_alphas_cumprod().to(device)
    schedule = make_schedule(n_steps, device)

    student = load_base_model(device)
    add_lora_heads(student, n_steps, args.rank, args.targets)
    if teacher_sched is not None:
        warm_start_from_teacher(student, schedule, teacher, teacher_sched)
    student.train()

    trainable = [p for p in student.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in student.parameters())
    print(f"  heads: {n_steps} | trainable LoRA params: {n_train / 1e6:.2f}M "
          f"({100 * n_train / n_total:.1f}% of {n_total / 1e6:.1f}M)")

    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    dataset = torchvision.datasets.CIFAR10(
        root=args.data_root, train=True, download=True,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True,
                        drop_last=True, num_workers=2)

    for epoch in range(args.epochs):
        pbar = tqdm(loader, desc=f"{n_steps}-step | epoch {epoch + 1}/{args.epochs}")
        for imgs, _ in pbar:
            imgs = imgs.to(device)

            # one jump (= one head) per batch; heads 0..n-2 are hit by jump losses
            i = int(torch.randint(0, n_steps - 1, (1,)))
            t_start, t_target = int(schedule[i]), int(schedule[i + 1])
            a_s = alphas[t_start].view(1, 1, 1, 1)
            a_e = alphas[t_target].view(1, 1, 1, 1)

            noise = torch.randn_like(imgs)
            x_t = a_s.sqrt() * imgs + (1 - a_s).sqrt() * noise

            # teacher: K DDIM sub-steps along [t_start, t_target], nearest head each
            with torch.no_grad():
                sub = torch.linspace(t_start, t_target, substeps + 1).round().long()
                x_ref = x_t.clone()
                for j in range(substeps):
                    eps = teacher_eps(teacher, teacher_sched, x_ref, int(sub[j]))
                    a_j = alphas[int(sub[j])].view(1, 1, 1, 1)
                    a_j1 = alphas[int(sub[j + 1])].view(1, 1, 1, 1)
                    x_ref = ddim_step(eps, x_ref, a_j, a_j1)

            # student: one jump with head i
            set_active_head(student, i)
            t_batch = torch.full((imgs.shape[0],), t_start, device=device, dtype=torch.long)
            s_eps = student(x_t, t_batch)
            x0_s = (x_t - (1 - a_s).sqrt() * s_eps) / a_s.sqrt()

            if loss_type == "huber_x0":                       # script-32 style (8-step stage)
                x0_ref = (x_ref - (1 - a_e).sqrt() * s_eps.detach()) / a_e.sqrt()
                loss = F.huber_loss(x0_s, x0_ref.detach(), delta=1.0)
            else:                                             # mse_position (scripts 30/31)
                x_fast = a_e.sqrt() * x0_s + (1 - a_e).sqrt() * s_eps
                loss = F.mse_loss(x_fast, x_ref.detach())

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            pbar.set_postfix({"head": i, "loss": f"{loss.item():.6f}"})

    return student, schedule


# ─────────────────────────────────────────────────────────────
# Evaluation (FID / IS / Precision / Recall via metrics_utils)
# ─────────────────────────────────────────────────────────────
def evaluate(model, schedule, args, device, label):
    from metrics_utils import prepare_real, compute_metrics   # lazy: needs torch-fidelity
    model.eval()
    alphas = linear_alphas_cumprod().to(device)
    print(f"\nEvaluating {label} on {args.eval_samples} samples…")
    extractor, real_imgs01, real_feats = prepare_real(args.eval_samples, args.data_root, device)

    fakes, done = [], 0
    while done < args.eval_samples:
        bs = min(args.eval_batch, args.eval_samples - done)
        fake01 = (generate_lora(model, schedule, alphas, bs, device) + 1.0) / 2.0
        fakes.append((fake01 * 255).round().byte().cpu())
        done += bs
        print(f"  generated {done}/{args.eval_samples}", end="\r")
    fakes = torch.cat(fakes, dim=0)

    m = compute_metrics(fakes, real_imgs01, real_feats, extractor, device,
                        args.eval_batch, args.knn)
    print(f"\n{label}: FID {m['fid']:.3f} | IS {m['is_mean']:.3f}±{m['is_std']:.3f} | "
          f"Precision {m['precision']:.3f} | Recall {m['recall']:.3f}")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--stages", type=int, nargs="+", default=[25, 12, 8],
                        help="step counts of the chained students (strictly decreasing)")
    parser.add_argument("--teacher", default=None,
                        help="LoRA checkpoint to act as the first stage's teacher "
                             "(default: the raw base model)")
    parser.add_argument("--rank", type=int, default=4, help="LoRA rank")
    parser.add_argument("--targets", nargs="+", default=["attn", "temb"],
                        choices=["attn", "temb", "conv"],
                        help="which layer groups get LoRA heads")
    parser.add_argument("--substeps", type=int, nargs="+", default=None,
                        help="teacher DDIM sub-steps per student jump, one per stage "
                             "(default: 4 2 3, mirroring scripts 30/31/32)")
    parser.add_argument("--loss", default=None, choices=["mse_position", "huber_x0"],
                        help="override the per-stage default (huber_x0 for <=8 steps, else mse)")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--outdir", default="./checkpoints")
    parser.add_argument("--eval-samples", type=int, default=0,
                        help="if >0, run FID/IS/Precision/Recall after each stage")
    parser.add_argument("--eval-batch", type=int, default=250)
    parser.add_argument("--knn", type=int, default=3)
    parser.add_argument("--eval-only", action="store_true",
                        help="skip training; evaluate --checkpoint")
    parser.add_argument("--checkpoint", default=None,
                        help="with --eval-only: the LoRA student checkpoint to evaluate")
    args = parser.parse_args()
    args.targets = set(args.targets)

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    print(f"Device: {device}")

    if args.eval_only:
        assert args.checkpoint, "--eval-only requires --checkpoint"
        if args.eval_samples <= 0:
            args.eval_samples = 10000
        model, schedule = load_lora_student(args.checkpoint, device)
        evaluate(model, schedule, args, device, label=os.path.basename(args.checkpoint))
        return

    assert all(a > b for a, b in zip(args.stages, args.stages[1:])), \
        "--stages must be strictly decreasing"
    default_sub = [4, 2, 3]
    substeps = args.substeps or [default_sub[i] if i < len(default_sub) else 2
                                 for i in range(len(args.stages))]
    assert len(substeps) == len(args.stages), "--substeps must match --stages in length"
    os.makedirs(args.outdir, exist_ok=True)

    if args.teacher:
        teacher, teacher_sched = load_lora_student(args.teacher, device)
    else:
        teacher, teacher_sched = load_base_model(device), None
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    for k, n_steps in enumerate(args.stages):
        loss_type = args.loss or ("huber_x0" if n_steps <= 8 else "mse_position")
        t_label = "base(1000)" if teacher_sched is None else f"{len(teacher_sched)}-step LoRA"
        print(f"\n=== Stage {k + 1}/{len(args.stages)}: {t_label} teacher -> "
              f"{n_steps}-step student ({substeps[k]} sub-steps, {loss_type}) ===")

        student, schedule = run_stage(n_steps, teacher, teacher_sched, substeps[k],
                                      loss_type, args, device, args.seed + k)

        path = os.path.join(args.outdir, f"lora_student_{n_steps}step.pt")
        torch.save({"n_steps": n_steps, "rank": args.rank, "targets": sorted(args.targets),
                    "schedule": schedule.cpu(), "lora_state": lora_state(student)}, path)
        print(f"  saved {path}")

        if args.eval_samples > 0:
            evaluate(student, schedule, args, device, label=f"{n_steps}-step LoRA student")

        teacher, teacher_sched = student, schedule            # chain to the next stage
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)


if __name__ == "__main__":
    main()
