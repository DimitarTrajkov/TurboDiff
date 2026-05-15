import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from diffusers import DDPMPipeline, DDIMScheduler, UNet2DModel
import copy
from tqdm import tqdm


import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet2DModel, DDPMPipeline, DDIMScheduler
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore


def run_final_eval(model, scheduler, dataloader, device, num_samples=10000, steps=8):
    print("\n--- Starting Final 10K Evaluation ---")
    model.eval()
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    is_metric = InceptionScore(normalize=True).to(device)
    
    # 1. Process Real Images in specific batches of 128
    real_count = 0
    # We use a smaller batch size here specifically for the GPU update to save memory
    eval_real_loader = DataLoader(dataloader.dataset, batch_size=128, shuffle=False)
    
    with torch.no_grad():
        for imgs, _ in tqdm(eval_real_loader, desc="Processing Real Images"):
            if real_count >= num_samples: 
                break
            # Move to device and normalize to [0, 1]
            # print(f"Processing real batch {real_count//128 + 1}...", end='\r')
            batch = (imgs.to(device) + 1.0) / 2.0
            fid.update(batch, real=True)
            real_count += batch.shape[0]

    # 2. Generate and Process Fake Images
    fake_count = 0
    alphas = scheduler.alphas_cumprod.to(device)
    timesteps = torch.linspace(scheduler.config.num_train_timesteps - 1, 0, steps, dtype=torch.long, device=device)
    
    with torch.no_grad():
        while fake_count < num_samples:
            # Using 100 or 128 for fake generation is usually safe for CIFAR-10
            print(f"Generating fake batch {fake_count//100 + 1}...", end='\r')
            curr_batch = min(100, num_samples - fake_count)
            
            x = torch.randn(curr_batch, 3, 32, 32, device=device)
            
            for i, t in enumerate(timesteps):
                t_batch = torch.full((curr_batch,), t, device=device, dtype=torch.long)
                noise_pred = model(x, t_batch).sample
                
                a_s = alphas[t].view(-1, 1, 1, 1)
                if i == len(timesteps) - 1:
                    x = (x - torch.sqrt(1 - a_s) * noise_pred) / torch.sqrt(a_s)
                    break
                    
                next_t = timesteps[i+1]
                a_e = alphas[next_t].view(-1, 1, 1, 1)
                x0_p = (x - torch.sqrt(1 - a_s) * noise_pred) / torch.sqrt(a_s)
                x = torch.sqrt(a_e) * x0_p + torch.sqrt(1 - a_e) * noise_pred
            
            samples = (x.clamp(-1, 1) + 1.0) / 2.0
            fid.update(samples, real=False)
            is_metric.update(samples)
            
            fake_count += curr_batch
            print(f"Generated {fake_count}/{num_samples} samples...", end='\r')

    print(f"\n[SURGICAL RESULTS]")
    print(f"FID: {fid.compute().item():.4f} | IS: {is_metric.compute()[0].item():.4f}")
    
    
    
    
def distill_to_8_steps():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load the 12-step Master
    pipe = DDPMPipeline.from_pretrained("google/ddpm-cifar10-32").to(device)
    teacher = UNet2DModel.from_config(pipe.unet.config).to(device)
    # Ensure this file exists in your directory
    teacher.load_state_dict(torch.load("fast_professor_12step.pt", map_location=device))
    teacher.eval()
    for p in teacher.parameters(): 
        p.requires_grad = False
    
    # 2. Create the 8-step Student
    student = copy.deepcopy(teacher).to(device)
    student.train()
    # Explicitly ensure student parameters require grad
    for p in student.parameters():
        p.requires_grad = True

    # Scheduler setup
    scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    scheduler.set_timesteps(24) 
    timesteps = scheduler.timesteps.to(device)
    alphas = scheduler.alphas_cumprod.to(device)

    optimizer = torch.optim.AdamW(student.parameters(), lr=8e-6)
    
    dataloader = DataLoader(
        CIFAR10(root='./data', train=True, download=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])), batch_size=32, shuffle=True, drop_last=True
    )

    print("\nStarting Progressive Distillation: 12 -> 8 Steps (3:1 Jump)...")
    for epoch in range(5):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/5")
        for imgs, _ in pbar:
            imgs = imgs.to(device)
            B = imgs.shape[0]
            noise = torch.randn_like(imgs)
            
            # Pick a starting point
            indices = torch.randint(0, len(timesteps) - 3, (B,), device=device)
            t0 = timesteps[indices]
            t1 = timesteps[indices + 1]
            t2 = timesteps[indices + 2]
            t3 = timesteps[indices + 3]
            
            # Use gather to get specific alphas for the batch
            a0 = alphas[t0].view(-1, 1, 1, 1)
            x_t = torch.sqrt(a0) * imgs + torch.sqrt(1 - a0) * noise

            # --- TEACHER: 3-STEP TRAJECTORY ---
            with torch.no_grad():
                # Step 1: t0 -> t1
                n1 = teacher(x_t, t0).sample
                x0_1 = (x_t - torch.sqrt(1 - a0) * n1) / torch.sqrt(a0)
                a1 = alphas[t1].view(-1, 1, 1, 1)
                x_1 = torch.sqrt(a1) * x0_1 + torch.sqrt(1 - a1) * n1
                
                # Step 2: t1 -> t2
                n2 = teacher(x_1, t1).sample
                x0_2 = (x_1 - torch.sqrt(1 - a1) * n2) / torch.sqrt(a1)
                a2 = alphas[t2].view(-1, 1, 1, 1)
                x_2 = torch.sqrt(a2) * x0_2 + torch.sqrt(1 - a2) * n2

                # Step 3: t2 -> t3
                n3 = teacher(x_2, t2).sample
                x0_3 = (x_2 - torch.sqrt(1 - a2) * n3) / torch.sqrt(a2)
                a3 = alphas[t3].view(-1, 1, 1, 1)
                x_ref = torch.sqrt(a3) * x0_3 + torch.sqrt(1 - a3) * n3

            # --- STUDENT: 1-STEP JUMP ---
            # IMPORTANT: x_t and s_n_pred are the only things that should carry grad here
            # --- STUDENT: LEARN THE SHORTCUT ---
            s_n_pred = student(x_t, t0).sample
            
            # 1. Derive the student's implied x0 (the "destination")
            # x0 = (x_t - sqrt(1-a0) * noise) / sqrt(a0)
            x0_student = (x_t - torch.sqrt(1 - a0) * s_n_pred) / torch.sqrt(a0)
            
            # 2. Derive the teacher's implied x0 from its 3-step result
            # This is what the student SHOULD have predicted to get to x_ref in one jump
            x0_teacher_target = (x_ref - torch.sqrt(1 - a3) * s_n_pred.detach()) / torch.sqrt(a3)

            # 3. Minimize the distance between their intended destinations
            # We use Huber loss (SmoothL1) as it's more robust to outliers in distillation
            loss = F.huber_loss(x0_student, x0_teacher_target.detach(), delta=1.0)
            
            optimizer.zero_grad()
            loss.backward()
            # Add gradient clipping to prevent the "boring" convergence
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            
            
            pbar.set_postfix({'loss': f"{loss.item():.7f}"})

    torch.save(student.state_dict(), "fast_professor_8step.pt")
    print("\n[EVALUATING 8-STEP STUDENT]")
    run_final_eval(student, scheduler, dataloader, device, num_samples=10000, steps=8)
    # run_final_eval(teacher, scheduler, dataloader, device, num_samples=10000, steps=8)

if __name__ == "__main__":
    distill_to_8_steps()
    
    
# STUDENT 8 step
# FID: 26.4792 | IS: 7.7493
# 12step-TEACHER 8 step
# FID: 26.7089 | IS: 7.8177


# Generated 10000/10000 samples...
# [SURGICAL RESULTS]
# FID: 15.6114 | IS: 8.7595