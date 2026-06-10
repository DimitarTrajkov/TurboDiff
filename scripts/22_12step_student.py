import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from diffusers import DDPMPipeline, DDIMScheduler, UNet2DModel
import copy
from tqdm import tqdm
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore

# ==========================================
# 1. Evaluation Helper
# ==========================================
@torch.no_grad()
def generate_fast_samples(unet, scheduler, batch_size, device, steps=12):
    unet.eval()
    # Linearly spaced steps for the new target speed
    timesteps = torch.linspace(scheduler.config.num_train_timesteps - 1, 0, steps, dtype=torch.long, device=device)
    x = torch.randn(batch_size, 3, 32, 32, device=device)
    alphas = scheduler.alphas_cumprod.to(device)
    
    for i, t in enumerate(timesteps):
        t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
        noise_pred = unet(x, t_batch).sample
        
        a_s = alphas[t].view(-1, 1, 1, 1)
        if i == len(timesteps) - 1:
            x = (x - torch.sqrt(1 - a_s) * noise_pred) / torch.sqrt(a_s)
            break
            
        next_t = timesteps[i + 1]
        a_e = alphas[next_t].view(-1, 1, 1, 1)
        x0_pred = (x - torch.sqrt(1 - a_s) * noise_pred) / torch.sqrt(a_s)
        x = torch.sqrt(a_e) * x0_pred + torch.sqrt(1 - a_e) * noise_pred
        
    return x.clamp(-1, 1)

def run_final_eval(model, scheduler, dataloader, device, num_samples=10000, steps=12):
    model.eval()
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    is_metric = InceptionScore(normalize=True).to(device)

    # Real Images
    real_count = 0
    for imgs, _ in tqdm(dataloader, desc="Feeding Real Images", leave=False):
        if real_count >= num_samples: break
        batch = (imgs.to(device) + 1.0) / 2.0
        fid.update(batch, real=True)
        real_count += batch.shape[0]

    # Fake Images
    fake_count = 0
    while fake_count < num_samples:
        curr_batch = min(64, num_samples - fake_count)
        samples = generate_fast_samples(model, scheduler, curr_batch, device, steps=steps)
        samples = (samples + 1.0) / 2.0
        fid.update(samples, real=False)
        is_metric.update(samples)
        fake_count += curr_batch
        
    print(f"--- Results ({steps} steps) ---")
    print(f"FID: {fid.compute().item():.4f} | IS: {is_metric.compute()[0].item():.4f}")

# ==========================================
# 2. Distillation: 25 Steps -> 12 Steps
# ==========================================
def halve_the_steps():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load the 25-step Teacher (Your previous best)
    pipe = DDPMPipeline.from_pretrained("google/ddpm-cifar10-32").to(device)
    teacher = UNet2DModel.from_config(pipe.unet.config).to(device)
    teacher.load_state_dict(torch.load("fast_professor_21_final.pt", map_location=device))
    teacher.eval()
    for p in teacher.parameters(): p.requires_grad = False
    
    # 2. Create the new 12-step Student (Starting from teacher weights)
    student = copy.deepcopy(teacher).to(device)
    student.train()
    for p in student.parameters(): p.requires_grad = True

    # Setup Scheduler (Based on the teacher's trained 25-step capability)
    scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    scheduler.set_timesteps(24) # We use a grid of 24 steps
    timesteps = scheduler.timesteps.to(device)
    alphas = scheduler.alphas_cumprod.to(device)

    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-5)
    
    dataset = CIFAR10(root='./data', train=True, download=True, 
                      transform=transforms.Compose([
                          transforms.RandomHorizontalFlip(),
                          transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                      ]))
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)

    print("\nStarting Progressive Distillation: Halving Steps (25 -> 12)...")
    for epoch in range(5):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/5")
        for imgs, _ in pbar:
            imgs = imgs.to(device)
            B = imgs.shape[0]
            noise = torch.randn_like(imgs)
            
            # Pick a starting point. We want to jump 2 steps of the teacher's schedule.
            indices = torch.randint(0, len(timesteps) - 2, (B,), device=device)
            t_start = timesteps[indices]
            t_mid = timesteps[indices + 1]
            t_target = timesteps[indices + 2]
            
            a_start = alphas[t_start].view(-1, 1, 1, 1)
            x_t = torch.sqrt(a_start) * imgs + torch.sqrt(1 - a_start) * noise

            # --- TEACHER (2 Steps) ---
            with torch.no_grad():
                # Step 1: x_t -> x_mid
                n1 = teacher(x_t, t_start).sample
                a_s = alphas[t_start].view(-1, 1, 1, 1)
                a_m = alphas[t_mid].view(-1, 1, 1, 1)
                x0_1 = (x_t - torch.sqrt(1 - a_s) * n1) / torch.sqrt(a_s)
                x_mid = torch.sqrt(a_m) * x0_1 + torch.sqrt(1 - a_m) * n1
                
                # Step 2: x_mid -> x_target
                n2 = teacher(x_mid, t_mid).sample
                a_e = alphas[t_target].view(-1, 1, 1, 1)
                x0_2 = (x_mid - torch.sqrt(1 - a_m) * n2) / torch.sqrt(a_m)
                x_ref = torch.sqrt(a_e) * x0_2 + torch.sqrt(1 - a_e) * n2

            # --- STUDENT (1 Jump) ---
            s_n_pred = student(x_t, t_start).sample
            a_e = alphas[t_target].view(-1, 1, 1, 1)
            x0_s = (x_t - torch.sqrt(1 - a_s) * s_n_pred) / torch.sqrt(a_s)
            x_fast = torch.sqrt(a_e) * x0_s + torch.sqrt(1 - a_e) * s_n_pred
            
            loss = F.mse_loss(x_fast, x_ref.detach())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix({'loss': f"{loss.item():.7f}"})

    # Save and Evaluate
    torch.save(student.state_dict(), "fast_professor_12step.pt")
    
    print("\n[EVALUATING NEW STUDENT AT 12 STEPS]")
    run_final_eval(student, scheduler, dataloader, device, num_samples=10000, steps=12)
    
    print("\n[EVALUATING PREVIOUS BEST AT 12 STEPS (FOR COMPARISON)]")
    run_final_eval(teacher, scheduler, dataloader, device, num_samples=10000, steps=12)

if __name__ == "__main__":
    halve_the_steps()
    
    
# [EVALUATING NEW STUDENT AT 12 STEPS]
# --- Results (12 steps) ---                                                                                                                                                             
# FID: 15.8880 | IS: 8.4327

# [EVALUATING PREVIOUS BEST AT 12 STEPS (FOR COMPARISON)]
# --- Results (12 steps) ---                                                                                                                                                             
# FID: 25.4690 | IS: 7.8701