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

# ==========================================
# 1. Hook Manager
# ==========================================
class HookTool:
    def __init__(self):
        self.fea = None
    def hook_fun(self, module, fea_in, fea_out):
        if isinstance(fea_out, tuple):
            self.fea = fea_out[0]
        else:
            self.fea = fea_out

# ==========================================
# 2. Evaluation Helper (12 Steps)
# ==========================================
@torch.no_grad()
def generate_fast_samples(unet, scheduler, batch_size, device, steps=12):
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
        next_t = timesteps[i+1]
        a_e = alphas[next_t].view(-1, 1, 1, 1)
        x0_pred = (x - torch.sqrt(1 - a_s) * noise_pred) / torch.sqrt(a_s)
        x = torch.sqrt(a_e) * x0_pred + torch.sqrt(1 - a_e) * noise_pred
    return x.clamp(-1, 1)

def run_final_eval(model, scheduler, dataloader, device, num_samples=10000, steps=12):
    print(f"\n--- PERFORMING 10K EVALUATION ({steps} steps) ---")
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
        curr_batch = min(100, num_samples - fake_count)
        samples = generate_fast_samples(model, scheduler, curr_batch, device, steps=steps)
        samples = (samples + 1.0) / 2.0
        fid.update(samples, real=False)
        is_metric.update(samples)
        fake_count += curr_batch
        
    print(f"\n[FINAL COMPRESSION RESULTS]")
    print(f"FID: {fid.compute().item():.4f}")
    print(f"IS:  {is_metric.compute()[0].item():.4f}")

# ==========================================
# 3. Main Training & Compression
# ==========================================
def train_compressed_junior():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Teacher Setup
    pipe = DDPMPipeline.from_pretrained("google/ddpm-cifar10-32").to(device)
    teacher = UNet2DModel.from_config(pipe.unet.config).to(device)
    teacher.load_state_dict(torch.load("fast_professor_12step.pt", map_location=device))
    teacher.eval()
    for p in teacher.parameters(): p.requires_grad = False

    # 2. Junior Student Setup (96, 192, 512, 512)
    student_config = teacher.config.copy()
    student_config["block_out_channels"] = (96, 192, 512, 512)
    student = UNet2DModel.from_config(student_config).to(device)
    student.train()

    # 3. Projectors for Feature Mismatch
    proj0 = nn.Conv2d(96, 128, kernel_size=1).to(device)
    proj1 = nn.Conv2d(192, 256, kernel_size=1).to(device)

    # 4. Symmetrical Hooks
    t_down = [HookTool() for _ in range(2)]; t_up = [HookTool() for _ in range(2)]
    s_down = [HookTool() for _ in range(2)]; s_up = [HookTool() for _ in range(2)]

    teacher.down_blocks[0].register_forward_hook(t_down[0].hook_fun)
    teacher.down_blocks[1].register_forward_hook(t_down[1].hook_fun)
    teacher.up_blocks[2].register_forward_hook(t_up[0].hook_fun) # 16x16
    teacher.up_blocks[3].register_forward_hook(t_up[1].hook_fun) # 32x32

    student.down_blocks[0].register_forward_hook(s_down[0].hook_fun)
    student.down_blocks[1].register_forward_hook(s_down[1].hook_fun)
    student.up_blocks[2].register_forward_hook(s_up[0].hook_fun)
    student.up_blocks[3].register_forward_hook(s_up[1].hook_fun)

    # 5. Optimization
    optimizer = torch.optim.AdamW(list(student.parameters()) + list(proj0.parameters()) + list(proj1.parameters()), lr=1e-4)
    scheduler_diff = DDIMScheduler.from_config(pipe.scheduler.config)
    scheduler_diff.set_timesteps(12)
    timesteps = scheduler_diff.timesteps.to(device)
    alphas = scheduler_diff.alphas_cumprod.to(device)

    dataloader = DataLoader(CIFAR10(root='./data', train=True, download=True, transform=transforms.Compose([
        transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])), batch_size=32, shuffle=True)

    print("\nStarting Junior Distillation (Structural Compression)...")
    for epoch in range(10):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/10")
        for imgs, _ in pbar:
            imgs = imgs.to(device)
            noise = torch.randn_like(imgs)
            idx = torch.randint(0, len(timesteps), (imgs.shape[0],), device=device)
            t = timesteps[idx]
            
            x_t = torch.sqrt(alphas[t]).view(-1,1,1,1) * imgs + torch.sqrt(1 - alphas[t]).view(-1,1,1,1) * noise

            with torch.no_grad():
                t_out = teacher(x_t, t).sample
                t_fs = [t_down[0].fea, t_down[1].fea, t_up[0].fea, t_up[1].fea]

            s_out = student(x_t, t).sample
            s_fs = [s_down[0].fea, s_down[1].fea, s_up[0].fea, s_up[1].fea]

            # Loss: Final Output + Downscale Mimicry + Upscale Mimicry
            loss_mse = F.mse_loss(s_out, t_out.detach())
            l_feat = (F.mse_loss(proj0(s_fs[0]), t_fs[0].detach()) + 
                      F.mse_loss(proj1(s_fs[1]), t_fs[1].detach()) +
                      F.mse_loss(proj1(s_fs[2]), t_fs[2].detach()) +
                      F.mse_loss(proj0(s_fs[3]), t_fs[3].detach()))
            
            total_loss = loss_mse + 0.1 * l_feat

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            pbar.set_postfix({'loss': f"{total_loss.item():.6f}"})

    # Save and Final Eval
    torch.save(student.state_dict(), "junior_professor_final.pt")
    run_final_eval(student, scheduler_diff, dataloader, device, num_samples=10000, steps=12)

if __name__ == "__main__":
    train_compressed_junior()
    
# Starting Junior Distillation (Structural Compression)...
# Epoch 1/10: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1563/1563 [08:44<00:00,  2.98it/s, loss=87.845200]
# Epoch 2/10: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1563/1563 [08:41<00:00,  3.00it/s, loss=92.116089]
# Epoch 3/10: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1563/1563 [08:37<00:00,  3.02it/s, loss=72.888466]
# Epoch 4/10: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1563/1563 [08:48<00:00,  2.96it/s, loss=63.335224]
# Epoch 5/10: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1563/1563 [08:49<00:00,  2.95it/s, loss=53.756332]
# Epoch 6/10: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1563/1563 [08:49<00:00,  2.95it/s, loss=24.774580]
# Epoch 7/10: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1563/1563 [08:43<00:00,  2.98it/s, loss=39.997334]
# Epoch 8/10: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1563/1563 [08:34<00:00,  3.04it/s, loss=54.782223]
# Epoch 9/10: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1563/1563 [08:42<00:00,  2.99it/s, loss=29.613119]
# Epoch 10/10: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1563/1563 [08:36<00:00,  3.02it/s, loss=49.505024]

# --- PERFORMING 10K EVALUATION (12 steps) ---
# C:\Users\Dimitar Trajkov\anaconda3\Lib\site-packages\torchmetrics\utilities\prints.py:43: UserWarning: Metric `InceptionScore` will save all extracted features in buffer. For large datasets this may lead to large memory footprint.
#   warnings.warn(*args, **kwargs)
                                                                                                                                                                                       
# [FINAL COMPRESSION RESULTS]
# FID: 205.1260
# IS:  3.7030