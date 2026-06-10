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
# 1. Hook Tool
# ==========================================
class HookTool:
    def __init__(self):
        self.fea = None
    def hook_fun(self, module, fea_in, fea_out):
        if isinstance(fea_out, tuple): self.fea = fea_out[0]
        else: self.fea = fea_out

# ==========================================
# 2. Main Training Script
# ==========================================

def run_final_eval(model, scheduler, dataloader, device, num_samples=10000, steps=12):
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
    
    
    
def train_frozen_brain_junior():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load Teacher (The Master)
    pipe = DDPMPipeline.from_pretrained("google/ddpm-cifar10-32").to(device)
    teacher = UNet2DModel.from_config(pipe.unet.config).to(device)
    teacher.load_state_dict(torch.load("fast_professor_12step.pt", map_location=device))
    teacher.eval()

    # 2. Create Junior (96-256-512-512)
    student_config = teacher.config.copy()
    student_config["block_out_channels"] = (96, 256, 512, 512)
    student = UNet2DModel.from_config(student_config).to(device)

    # 3. SELECTIVE WARM START & FREEZING
    print("Freezing the 'Brain' (256/512 layers) and warming up matching weights...")
    t_state = teacher.state_dict()
    s_state = student.state_dict()
    
    for name, param in student.named_parameters():
        # If the layer exists in teacher and matches shape, COPY and FREEZE
        if name in t_state and param.shape == t_state[name].shape:
            param.data.copy_(t_state[name])
            if "down_blocks.0" not in name and "up_blocks.3" not in name:
                param.requires_grad = False # Freeze mid-layers
        else:
            # These are the 96-channel layers - leave them trainable
            param.requires_grad = True

    # 4. Hooks & Projectors
    t_h_down, t_h_up = HookTool(), HookTool()
    s_h_down, s_h_up = HookTool(), HookTool()
    teacher.down_blocks[0].register_forward_hook(t_h_down.hook_fun)
    teacher.up_blocks[3].register_forward_hook(t_h_up.hook_fun)
    student.down_blocks[0].register_forward_hook(s_h_down.hook_fun)
    student.up_blocks[3].register_forward_hook(s_h_up.hook_fun)
    
    # Projectors to bridge 96 -> 128
    proj_d = nn.Conv2d(96, 128, kernel_size=1).to(device)
    proj_u = nn.Conv2d(96, 128, kernel_size=1).to(device)

    # Only optimize trainable parameters
    optimizer = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad] + 
        list(proj_d.parameters()) + list(proj_u.parameters()), 
        lr=2e-4 # Higher LR since most of the model is frozen
    )

    # Setup Scheduler & Data
    scheduler_diff = DDIMScheduler.from_config(pipe.scheduler.config)
    scheduler_diff.set_timesteps(12)
    timesteps, alphas = scheduler_diff.timesteps.to(device), scheduler_diff.alphas_cumprod.to(device)
    
    dataloader = DataLoader(CIFAR10(root='./data', train=True, download=True, transform=transforms.Compose([
        transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])), batch_size=32, shuffle=True)

    print("\nTraining 'Eyes' (96ch) while 'Brain' (256/512ch) is frozen...")
    for epoch in range(15): # 15 epochs to let the eyes adjust
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/15")
        for imgs, _ in pbar:
            imgs = imgs.to(device); noise = torch.randn_like(imgs)
            t = timesteps[torch.randint(0, len(timesteps), (imgs.shape[0],), device=device)]
            x_t = torch.sqrt(alphas[t]).view(-1,1,1,1) * imgs + torch.sqrt(1 - alphas[t]).view(-1,1,1,1) * noise

            with torch.no_grad():
                t_out = teacher(x_t, t).sample
                t_fd, t_fu = t_h_down.fea, t_h_up.fea

            s_out = student(x_t, t).sample
            s_fd, s_fu = s_h_down.fea, s_h_up.fea

            loss_mse = F.mse_loss(s_out, t_out.detach())
            
            # Normalize feature loss to avoid the '54.0' spike
            loss_f_d = F.mse_loss(proj_d(s_fd), t_fd.detach()) / (t_fd.detach().pow(2).mean() + 1e-6)
            loss_f_u = F.mse_loss(proj_u(s_fu), t_fu.detach()) / (t_fu.detach().pow(2).mean() + 1e-6)
            loss_feat = loss_f_d + loss_f_u
            
            total_loss = loss_mse + 0.2 * loss_feat

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            pbar.set_postfix({'mse': f"{loss_mse.item():.4f}", 'feat': f"{loss_feat.item():.4f}"})

    torch.save(student.state_dict(), "junior_frozen_brain_96.pt")
    run_final_eval(student, scheduler_diff, dataloader, device)

# [Include the VRAM-friendly run_final_eval function from previous responses here]

if __name__ == "__main__":
    train_frozen_brain_junior()