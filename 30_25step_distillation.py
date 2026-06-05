"""
30_25step_distillation.py

Pure-PyTorch equivalent of script 21.
Progressive distillation: Google DDPM (1000 steps) → 25 steps.

Training grid : 100 DDIM steps  (step size = 10 timestep units)
Teacher jump  : 4 grid steps → student learns to match in 1 step
Output        : fast_professor_21_final.pt
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
    generate_n_steps,
    run_eval,
    WEIGHTS_PATH,
)

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
TEACHER_JUMP = 4     # teacher takes 4 DDIM steps; student learns to match in 1
GRID_STEPS   = 100   # training timestep grid size
EVAL_STEPS   = 25    # inference steps at evaluation time (100 // 4 = 25)
EPOCHS       = 5
LR           = 1e-5
BATCH_SIZE   = 32


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

    # Frozen teacher = original Google DDPM weights
    teacher = UNet2DModel().to(DEVICE)
    teacher.load_state_dict(
        torch.load(WEIGHTS_PATH, map_location=DEVICE, weights_only=True), strict=True
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

    print(f"\nProgressive distillation: 1000 → {EVAL_STEPS} steps")
    print(f"Grid: {GRID_STEPS} steps, teacher jump: {TEACHER_JUMP} grid steps\n")

    for epoch in range(EPOCHS):
        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")
        for imgs, _ in pbar:
            imgs  = imgs.to(DEVICE)
            B     = imgs.shape[0]
            noise = torch.randn_like(imgs)

            # Random starting positions leaving room for TEACHER_JUMP steps ahead
            idx     = torch.randint(0, len(timesteps) - TEACHER_JUMP, (B,), device=DEVICE)
            t_start = timesteps[idx]
            t_target = timesteps[idx + TEACHER_JUMP]

            a_start = alphas_cumprod[t_start].view(-1, 1, 1, 1)
            x_t     = a_start.sqrt() * imgs + (1 - a_start).sqrt() * noise

            # Teacher: TEACHER_JUMP sequential DDIM steps (no grad)
            with torch.no_grad():
                x_ref = x_t.clone()
                for i in range(TEACHER_JUMP):
                    t_curr = timesteps[idx + i]
                    t_next = timesteps[idx + i + 1]
                    eps    = teacher(x_ref, t_curr)
                    a_s    = alphas_cumprod[t_curr].view(-1, 1, 1, 1)
                    a_e    = alphas_cumprod[t_next].view(-1, 1, 1, 1)
                    x0     = (x_ref - (1 - a_s).sqrt() * eps) / a_s.sqrt()
                    x_ref  = a_e.sqrt() * x0 + (1 - a_e).sqrt() * eps

            # Student: 1 DDIM step from t_start directly to t_target
            s_eps   = student(x_t, t_start)
            a_s     = alphas_cumprod[t_start].view(-1, 1, 1, 1)
            a_e     = alphas_cumprod[t_target].view(-1, 1, 1, 1)
            x0_s    = (x_t - (1 - a_s).sqrt() * s_eps) / a_s.sqrt()
            x_fast  = a_e.sqrt() * x0_s + (1 - a_e).sqrt() * s_eps

            loss = F.mse_loss(x_fast, x_ref.detach())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix({"loss": f"{loss.item():.7f}"})

    torch.save(student.state_dict(), "fast_professor_21_final.pt")
    print("\nSaved fast_professor_21_final.pt")

    print(f"\n[Student at {EVAL_STEPS} steps]")
    run_eval(student, alphas_cumprod.cpu(), dataset, DEVICE,
             num_samples=10000, steps=EVAL_STEPS)

    print(f"\n[Teacher (original DDPM) at {EVAL_STEPS} steps — reference]")
    run_eval(teacher, alphas_cumprod.cpu(), dataset, DEVICE,
             num_samples=10000, steps=EVAL_STEPS)


if __name__ == "__main__":
    distill()

# Expected results (from original script 21):
# Student  25 steps: FID 15.91 | IS 8.34
# Teacher  25 steps: FID 21.65 | IS 7.99
