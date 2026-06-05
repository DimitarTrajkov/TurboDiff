"""
32_8step_distillation.py

Pure-PyTorch equivalent of script 27.
Progressive distillation: 12 steps → 8 steps.

Loads the 12-step model (fast_professor_12step.pt) as the frozen teacher.
Training grid : 24 DDIM steps
Teacher jump  : 3 grid steps → student matches in 1 step
Loss          : Huber loss on x0 predictions (different from scripts 30/31)
Output        : fast_professor_8step.pt
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
)

DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
TEACHER_JUMP = 3     # teacher takes 3 DDIM steps; student matches in 1
GRID_STEPS   = 24    # training timestep grid size
EVAL_STEPS   = 8     # inference steps at evaluation
EPOCHS       = 5
LR           = 8e-6  # lower than previous rounds
BATCH_SIZE   = 32
GRAD_CLIP    = 1.0

TEACHER_WEIGHTS = "fast_professor_12step.pt"
STUDENT_WEIGHTS = "fast_professor_8step.pt"


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

    # Teacher: 12-step distilled model
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

    print(f"\nProgressive distillation: 12 → {EVAL_STEPS} steps")
    print(f"Grid: {GRID_STEPS} steps, teacher jump: {TEACHER_JUMP} grid steps\n")

    for epoch in range(EPOCHS):
        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")
        for imgs, _ in pbar:
            imgs  = imgs.to(DEVICE)
            B     = imgs.shape[0]
            noise = torch.randn_like(imgs)

            idx = torch.randint(0, len(timesteps) - TEACHER_JUMP, (B,), device=DEVICE)
            t0  = timesteps[idx]
            t1  = timesteps[idx + 1]
            t2  = timesteps[idx + 2]
            t3  = timesteps[idx + 3]

            a0  = alphas_cumprod[t0].view(-1, 1, 1, 1)
            x_t = a0.sqrt() * imgs + (1 - a0).sqrt() * noise

            # Teacher: 3 sequential DDIM steps (no grad)
            with torch.no_grad():
                # t0 → t1
                eps1 = teacher(x_t, t0)
                a1   = alphas_cumprod[t1].view(-1, 1, 1, 1)
                x0_1 = (x_t - (1 - a0).sqrt() * eps1) / a0.sqrt()
                x_1  = a1.sqrt() * x0_1 + (1 - a1).sqrt() * eps1

                # t1 → t2
                eps2 = teacher(x_1, t1)
                a2   = alphas_cumprod[t2].view(-1, 1, 1, 1)
                x0_2 = (x_1 - (1 - a1).sqrt() * eps2) / a1.sqrt()
                x_2  = a2.sqrt() * x0_2 + (1 - a2).sqrt() * eps2

                # t2 → t3  (x_ref = where the teacher ends up)
                eps3  = teacher(x_2, t2)
                a3    = alphas_cumprod[t3].view(-1, 1, 1, 1)
                x0_3  = (x_2 - (1 - a2).sqrt() * eps3) / a2.sqrt()
                x_ref = a3.sqrt() * x0_3 + (1 - a3).sqrt() * eps3

            # Student: 1-step prediction from t0
            s_eps = student(x_t, t0)

            # Student's x0 prediction from t0
            x0_student = (x_t - (1 - a0).sqrt() * s_eps) / a0.sqrt()

            # Teacher's implied x0 inferred from x_ref, using s_eps as the noise direction.
            # This asks: "what x0 would give x_ref if the noise direction were s_eps?"
            x0_teacher_target = (x_ref - (1 - a3).sqrt() * s_eps.detach()) / a3.sqrt()

            loss = F.huber_loss(x0_student, x0_teacher_target.detach(), delta=1.0)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), GRAD_CLIP)
            optimizer.step()
            pbar.set_postfix({"loss": f"{loss.item():.7f}"})

    torch.save(student.state_dict(), STUDENT_WEIGHTS)
    print(f"\nSaved {STUDENT_WEIGHTS}")

    print(f"\n[8-step student]")
    run_eval(student, alphas_cumprod.cpu(), dataset, DEVICE,
             num_samples=10000, steps=EVAL_STEPS)


if __name__ == "__main__":
    distill()

# Expected results (from original script 27):
# Student  8 steps: FID 26.48 | IS 7.75
# Teacher  8 steps: FID 26.71 | IS 7.82
# Best run  8 steps: FID 15.61 | IS 8.76
