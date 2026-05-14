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
# 1. Surgical Weight Transfer Logic
# ==========================================
def surgical_transfer(teacher, student):
    t_state = teacher.state_dict()
    s_state = student.state_dict()

    for name, s_param in s_state.items():
        if name in t_state:
            t_param = t_state[name]
            
            # 1. Perfect match: Direct Copy
            if s_param.shape == t_param.shape:
                s_param.copy_(t_param)
                continue
            
            # 2. Mismatch handling
            try:
                # Handle 4D weights (Conv layers)
                if len(s_param.shape) == 4:
                    # Output channels mismatch (Dim 0)
                    if s_param.shape[0] == 96 and t_param.shape[0] == 128:
                        s_param[:64].copy_(t_param[:64])
                        avg = t_param[64:128].reshape(32, 2, *t_param.shape[1:]).mean(dim=1)
                        s_param[64:96].copy_(avg)
                    
                    # Input channels mismatch (Dim 1)
                    elif s_param.shape[1] == 96 and t_param.shape[1] == 128:
                        s_param[:, :64].copy_(t_param[:, :64])
                        avg = t_param[:, 64:128].reshape(t_param.shape[0], 32, 2, *t_param.shape[2:]).mean(dim=2)
                        s_param[:, 64:96].copy_(avg)

                # Handle 1D tensors (Normalization weights, biases, scale-shift)
                elif len(s_param.shape) == 1:
                    if s_param.shape[0] == 96 and t_param.shape[0] == 128:
                        s_param[:64].copy_(t_param[:64])
                        avg = t_param[64:128].reshape(32, 2).mean(dim=1)
                        s_param[64:96].copy_(avg)
                
                # Handle 2D tensors (Linear layers if any)
                elif len(s_param.shape) == 2:
                    if s_param.shape[0] == 96 and t_param.shape[0] == 128:
                        s_param[:64].copy_(t_param[:64])
                        avg = t_param[64:128].reshape(32, 2, -1).mean(dim=1)
                        s_param[64:96].copy_(avg.squeeze())

                print(f"Successfully performed surgery on: {name}")

            except Exception as e:
                print(f"Skipping surgery for {name} due to shape logic: {e}")
                # Fallback: just copy what we can to avoid crashing
                min_dim0 = min(s_param.shape[0], t_param.shape[0])
                if len(s_param.shape) > 1:
                    min_dim1 = min(s_param.shape[1], t_param.shape[1])
                    s_param[:min_dim0, :min_dim1].copy_(t_param[:min_dim0, :min_dim1])
                else:
                    s_param[:min_dim0].copy_(t_param[:min_dim0])

    student.load_state_dict(s_state)
    return student

# ==========================================
# 2. Main Training Script
# ==========================================
def train_surgical_junior():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load Teacher
    pipe = DDPMPipeline.from_pretrained("google/ddpm-cifar10-32").to(device)
    teacher = UNet2DModel.from_config(pipe.unet.config).to(device)
    teacher.load_state_dict(torch.load("fast_professor_12step.pt", map_location=device))
    teacher.eval()

    # Create Student (96, 256, 512, 512)
    student_config = teacher.config.copy()
    student_config["block_out_channels"] = (96, 256, 512, 512)
    student = UNet2DModel.from_config(student_config).to(device)
    
    # PERFORM SURGERY
    print("Performing Surgical Weight Transfer...")
    student = surgical_transfer(teacher, student)
    student.train()

    optimizer = torch.optim.AdamW(student.parameters(), lr=5e-5) # Lower LR for surgery fine-tuning
    scheduler_diff = DDIMScheduler.from_config(pipe.scheduler.config)
    scheduler_diff.set_timesteps(12)
    timesteps, alphas = scheduler_diff.timesteps.to(device), scheduler_diff.alphas_cumprod.to(device)

    dataloader = DataLoader(CIFAR10(root='./data', train=True, download=True, transform=transforms.Compose([
        transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])), batch_size=32, shuffle=True)

    # print("\nStarting Junior Distillation (Surgical 96-256-512-512)...")
    # for epoch in range(5): # Fewer epochs needed because of surgery
    #     pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/5")
    #     for imgs, _ in pbar:
    #         imgs = imgs.to(device); noise = torch.randn_like(imgs)
    #         t = timesteps[torch.randint(0, len(timesteps), (imgs.shape[0],), device=device)]
    #         x_t = torch.sqrt(alphas[t]).view(-1,1,1,1) * imgs + torch.sqrt(1 - alphas[t]).view(-1,1,1,1) * noise

    #         with torch.no_grad():
    #             t_out = teacher(x_t, t).sample

    #         s_out = student(x_t, t).sample
    #         loss = F.mse_loss(s_out, t_out.detach())

    #         optimizer.zero_grad()
    #         loss.backward()
    #         torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
    #         optimizer.step()
    #         pbar.set_postfix({'img_mse': f"{loss.item():.6f}"})

    # torch.save(student.state_dict(), "junior_surgical_96.pt")
    student.load_state_dict(torch.load("junior_surgical_96.pt", map_location=device, weights_only=True))

    # 10K EVALUATION
    run_final_eval(student, scheduler_diff, dataloader, device)

# [Generic Eval Function provided in previous steps goes here]
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

if __name__ == "__main__":
    train_surgical_junior()

# Starting Junior Distillation (Surgical 96-256-512-512)...
# Epoch 1/5: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1563/1563 [09:42<00:00,  2.68it/s, img_mse=0.022300]
# Epoch 2/5: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1563/1563 [09:30<00:00,  2.74it/s, img_mse=0.009905]
# Epoch 3/5: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1563/1563 [09:31<00:00,  2.74it/s, img_mse=0.003714]
# Epoch 4/5: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1563/1563 [09:31<00:00,  2.74it/s, img_mse=0.005589]
# Epoch 5/5: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1563/1563 [09:37<00:00,  2.70it/s, img_mse=0.003836]
# [SURGICAL RESULTS]
# FID: 179.5734 | IS: 3.6789