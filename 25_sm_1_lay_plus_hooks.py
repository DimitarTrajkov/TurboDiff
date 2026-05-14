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
# 2. Bulletproof Surgery Logic
# ==========================================
def surgical_transfer(teacher, student):
    t_state = teacher.state_dict()
    s_state = student.state_dict()
    
    for name, s_param in s_state.items():
        if name not in t_state:
            continue
        t_param = t_state[name]
        
        # Scenario A: Shapes match perfectly
        if s_param.shape == t_param.shape:
            s_param.copy_(t_param)
            continue
        
        # Scenario B: Surgery required
        try:
            with torch.no_grad():
                # 4D Weights (Conv Layers)
                if len(s_param.shape) == 4:
                    # Case 1: Output channels changed (128 -> 96)
                    if s_param.shape[0] == 96 and t_param.shape[0] == 128:
                        s_param[:64].copy_(t_param[:64])
                        avg = t_param[64:128].reshape(32, 2, *t_param.shape[1:]).mean(dim=1)
                        s_param[64:96].copy_(avg)
                    # Case 2: Input channels changed (receiving 96 instead of 128)
                    elif s_param.shape[1] == 96 and t_param.shape[1] == 128:
                        s_param[:, :64].copy_(t_param[:, :64])
                        avg = t_param[:, 64:128].reshape(t_param.shape[0], 32, 2, *t_param.shape[2:]).mean(dim=2)
                        s_param[:, 64:96].copy_(avg)
                
                # 1D Weights (Norms, Biases, Time Embeddings)
                elif len(s_param.shape) == 1:
                    if s_param.shape[0] == 96 and t_param.shape[0] == 128:
                        s_param[:64].copy_(t_param[:64])
                        avg = t_param[64:128].reshape(32, 2).mean(dim=1)
                        s_param[64:96].copy_(avg)
                
                # Fallback for weird layers: Copy minimum common slice
                else:
                    slices = [slice(0, min(s, t)) for s, t in zip(s_param.shape, t_param.shape)]
                    s_param[slices].copy_(t_param[slices])
                    
        except Exception as e:
            print(f"Skipping surgery for {name}: {e}")
            
    return student

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
    
# ==========================================
# 3. Main Training
# ==========================================
def train_full_hybrid_junior():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Setup
    pipe = DDPMPipeline.from_pretrained("google/ddpm-cifar10-32").to(device)
    teacher = UNet2DModel.from_config(pipe.unet.config).to(device)
    teacher.load_state_dict(torch.load("fast_professor_12step.pt", map_location=device))
    teacher.eval()

    student_config = teacher.config.copy()
    student_config["block_out_channels"] = (96, 256, 512, 512)
    student = UNet2DModel.from_config(student_config).to(device)
    
    print("Performing Surgical Initialization...")
    student = surgical_transfer(teacher, student)
    student.train()

    # Hooks & Projectors
    t_h_down, t_h_up = HookTool(), HookTool()
    s_h_down, s_h_up = HookTool(), HookTool()
    teacher.down_blocks[0].register_forward_hook(t_h_down.hook_fun)
    teacher.up_blocks[3].register_forward_hook(t_h_up.hook_fun)
    student.down_blocks[0].register_forward_hook(s_h_down.hook_fun)
    student.up_blocks[3].register_forward_hook(s_h_up.hook_fun)
    
    # 1x1 Convs to bridge the 96 -> 128 gap for feature matching
    proj_d = nn.Conv2d(96, 128, kernel_size=1).to(device)
    proj_u = nn.Conv2d(96, 128, kernel_size=1).to(device)

    optimizer = torch.optim.AdamW(list(student.parameters()) + list(proj_d.parameters()) + list(proj_u.parameters()), lr=1e-4)
    scheduler_diff = DDIMScheduler.from_config(pipe.scheduler.config)
    scheduler_diff.set_timesteps(12)
    timesteps, alphas = scheduler_diff.timesteps.to(device), scheduler_diff.alphas_cumprod.to(device)

    dataloader = DataLoader(CIFAR10(root='./data', train=True, download=True, transform=transforms.Compose([
        transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])), batch_size=32, shuffle=True)

    print("\nStarting Hybrid Distillation...")
    for epoch in range(10):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/10")
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
            # loss_feat = F.mse_loss(proj_d(s_fd), t_fd.detach()) + F.mse_loss(proj_u(s_fu), t_fu.detach())
            # Normalize feature loss so it's on a similar scale to MSE
            loss_feat = (F.mse_loss(proj_d(s_fd), t_fd.detach()) / t_fd.detach().pow(2).mean()) + \
                        (F.mse_loss(proj_u(s_fu), t_fu.detach()) / t_fu.detach().pow(2).mean())
                        
            total_loss = loss_mse + 0.1 * loss_feat # Feature weight lowered to 0.1 for stability

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            pbar.set_postfix({'mse': f"{loss_mse.item():.4f}", 'feat': f"{loss_feat.item():.4f}"})

    torch.save(student.state_dict(), "junior_final_hooks.pt")
    run_final_eval(student, scheduler_diff, dataloader, device)
    # run_final_eval (VRAM-friendly version) here
    
if __name__ == "__main__":
    train_full_hybrid_junior()