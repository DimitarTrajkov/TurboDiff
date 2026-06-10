import torch
import torch.nn as nn
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
# 1. Evaluation Helpers
# ==========================================
@torch.no_grad()
def generate_fast_samples(unet, scheduler, batch_size, device, steps=25):
    unet.eval()
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

def run_final_eval(model, scheduler, dataloader, device, num_samples=10000, steps=25):
    print(f"\n--- FINAL EVALUATION: {num_samples} samples at {steps} steps ---")
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    is_metric = InceptionScore(normalize=True).to(device)

    # Real Images
    real_count = 0
    for imgs, _ in tqdm(dataloader, desc="Feeding Real Images"):
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
        
    print(f"FID: {fid.compute().item():.4f} | IS: {is_metric.compute()[0].item():.4f}")

# ==========================================
# 2. Distillation Logic
# ==========================================
def distill_professor():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load Teacher
    pipe = DDPMPipeline.from_pretrained("google/ddpm-cifar10-32").to(device)
    professor = pipe.unet
    professor.eval()
    for p in professor.parameters(): p.requires_grad = False
    
    # Fast Professor
    fast_prof = UNet2DModel.from_config(professor.config).to(device)
    fast_prof.load_state_dict(professor.state_dict())
    fast_prof.train()
    # FORCE GRADIENTS ON
    for p in fast_prof.parameters(): p.requires_grad = True

    scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    scheduler.set_timesteps(100)
    timesteps = scheduler.timesteps.to(device)
    alphas = scheduler.alphas_cumprod.to(device)

    optimizer = torch.optim.AdamW(fast_prof.parameters(), lr=1e-5)
    
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)

    # print("\nStarting Fast Professor Training...")
    # for epoch in range(5):
    #     pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/5")
    #     for imgs, _ in pbar:
    #         imgs = imgs.to(device)
    #         noise = torch.randn_like(imgs)
    #         indices = torch.randint(0, len(timesteps) - 4, (imgs.shape[0],), device=device)
            
    #         t_start = timesteps[indices]
    #         t_target = timesteps[indices + 4]
    #         a_start = alphas[t_start].view(-1, 1, 1, 1)
    #         x_t = torch.sqrt(a_start) * imgs + torch.sqrt(1 - a_start) * noise

    #         # 1. Teacher (NO GRAD)
    #         with torch.no_grad():
    #             x_ref = x_t.clone()
    #             for i in range(4):
    #                 curr_t_idx = timesteps[indices + i]
    #                 next_t_idx = timesteps[indices + i + 1]
    #                 n_teacher = professor(x_ref, curr_t_idx).sample
                    
    #                 a_s = alphas[curr_t_idx].view(-1, 1, 1, 1)
    #                 a_e = alphas[next_t_idx].view(-1, 1, 1, 1)
    #                 x0_p = (x_ref - torch.sqrt(1 - a_s) * n_teacher) / torch.sqrt(a_s)
    #                 x_ref = torch.sqrt(a_e) * x0_p + torch.sqrt(1 - a_e) * n_teacher

    #         # 2. Student (WITH GRAD)
    #         # Ensure this is NOT under a no_grad block
    #         s_n_pred = fast_prof(x_t, t_start).sample
            
    #         a_s = alphas[t_start].view(-1, 1, 1, 1)
    #         a_e = alphas[t_target].view(-1, 1, 1, 1)
    #         x0_s = (x_t - torch.sqrt(1 - a_s) * s_n_pred) / torch.sqrt(a_s)
    #         x_fast = torch.sqrt(a_e) * x0_s + torch.sqrt(1 - a_e) * s_n_pred
            
    #         loss = F.mse_loss(x_fast, x_ref)
            
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #         # pbar.set_postfix({'loss': f"{loss.item():.7f}"})
            
    # torch.save(fast_prof.state_dict(), "fast_professor_21_final.pt")
    run_final_eval(professor, scheduler, dataloader, device, num_samples=10000)

if __name__ == "__main__":
    distill_professor()
    
# the student 10K samples 25 steps: FID: 15.9068 | IS: 8.3419 
# the professor 10K samples 25 steps: FID: 21.6506 | IS: 7.9916