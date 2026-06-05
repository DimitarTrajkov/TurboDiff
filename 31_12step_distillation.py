"""
31_12step_distillation.py

Pure-PyTorch equivalent of script 22.
Progressive distillation: 25 steps → 12 steps.

Loads the 25-step model (fast_professor_21_final.pt) as the frozen teacher.
Training grid : 24 DDIM steps  (step size ≈ 41 timestep units)
Teacher jump  : 2 grid steps → student matches in 1 step
Output        : fast_professor_12step.pt
"""

import copy
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from tqdm import tqdm

from ddpm_arch import (
    UNet2DModel,
    linear_alphas_cumprod,
    make_training_grid,
    run_eval,
    WEIGHTS_PATH,
)

DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
TEACHER_JUMP = 2     # teacher takes 2 DDIM steps; student matches in 1
GRID_STEPS   = 24    # training timestep grid size
EVAL_STEPS   = 12    # inference steps at evaluation (24 // 2 = 12)
EPOCHS       = 5
LR           = 1e-5
BATCH_SIZE   = 32

TEACHER_WEIGHTS = "fast_professor_21_final.pt"
STUDENT_WEIGHTS = "fast_professor_12step.pt"


def build_dataset():
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    return CIFAR10(root="./data", train=True, download=True, transform=transform)


def distill():
    alphas_cumprod = linear_alphas_cumprod().to(DEVICE)
    timesteps      = make_training_grid(GRID_STEPS).to(DEVICE)

    # Teacher: 25-step distilled model
    teacher = UNet2DModel().to(DEVICE)
    teacher.load_state_dict(
        torch.load(TEACHER_WEIGHTS, map_location=DEVICE, weights_only=True), strict=True
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # Student starts as a perfect clone of the teacher
    student = copy.deepcopy(teacher).to(DEVICE)
    student.train()
    for p in student.parameters():
        p.requires_grad = True

    optimizer = torch.optim.AdamW(student.parameters(), lr=LR)
    dataset   = build_dataset()
    loader    = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    print(f"\nProgressive distillation: 25 → {EVAL_STEPS} steps")
    print(f"Grid: {GRID_STEPS} steps, teacher jump: {TEACHER_JUMP} grid steps\n")

    for epoch in range(EPOCHS):
        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")
        for imgs, _ in pbar:
            imgs  = imgs.to(DEVICE)
            B     = imgs.shape[0]
            noise = torch.randn_like(imgs)

            idx      = torch.randint(0, len(timesteps) - TEACHER_JUMP, (B,), device=DEVICE)
            t_start  = timesteps[idx]
            t_mid    = timesteps[idx + 1]
            t_target = timesteps[idx + TEACHER_JUMP]

            a_start = alphas_cumprod[t_start].view(-1, 1, 1, 1)
            x_t     = a_start.sqrt() * imgs + (1 - a_start).sqrt() * noise

            # Teacher: 2 DDIM steps (no grad)
            with torch.no_grad():
                # Step 1: t_start → t_mid
                eps1  = teacher(x_t, t_start)
                a_s   = alphas_cumprod[t_start].view(-1, 1, 1, 1)
                a_m   = alphas_cumprod[t_mid].view(-1, 1, 1, 1)
                x0_1  = (x_t - (1 - a_s).sqrt() * eps1) / a_s.sqrt()
                x_mid = a_m.sqrt() * x0_1 + (1 - a_m).sqrt() * eps1

                # Step 2: t_mid → t_target
                eps2   = teacher(x_mid, t_mid)
                a_e    = alphas_cumprod[t_target].view(-1, 1, 1, 1)
                x0_2   = (x_mid - (1 - a_m).sqrt() * eps2) / a_m.sqrt()
                x_ref  = a_e.sqrt() * x0_2 + (1 - a_e).sqrt() * eps2

            # Student: 1 DDIM step from t_start to t_target
            s_eps  = student(x_t, t_start)
            a_s    = alphas_cumprod[t_start].view(-1, 1, 1, 1)
            a_e    = alphas_cumprod[t_target].view(-1, 1, 1, 1)
            x0_s   = (x_t - (1 - a_s).sqrt() * s_eps) / a_s.sqrt()
            x_fast = a_e.sqrt() * x0_s + (1 - a_e).sqrt() * s_eps

            loss = F.mse_loss(x_fast, x_ref.detach())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix({"loss": f"{loss.item():.7f}"})

    torch.save(student.state_dict(), STUDENT_WEIGHTS)
    print(f"\nSaved {STUDENT_WEIGHTS}")

    print(f"\n[New student at {EVAL_STEPS} steps]")
    run_eval(student, alphas_cumprod.cpu(), dataset, DEVICE,
             num_samples=10000, steps=EVAL_STEPS)

    print(f"\n[Previous teacher at {EVAL_STEPS} steps — reference]")
    run_eval(teacher, alphas_cumprod.cpu(), dataset, DEVICE,
             num_samples=10000, steps=EVAL_STEPS)


if __name__ == "__main__":
    distill()

# Expected results (from original script 22):
# Student  12 steps: FID 15.89 | IS 8.43
# Teacher  12 steps: FID 25.47 | IS 7.87
