import torch
import time
import os
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as vutils

def plot_generated_images(model, vae, batch_size=32, steps=8):
    # 1. Run the inference
    imgs, time_taken = measure_inference(batch_size=batch_size, steps=steps)
    
    # 2. De-normalize: Tanh (-1 to 1) -> (0 to 1)
    imgs = (imgs * 0.5 + 0.5).clamp(0, 1).cpu()
    
    # 3. Create a grid
    grid = vutils.make_grid(imgs, nrow=8, padding=2, normalize=False)
    
    # 4. Convert to numpy for plotting
    plt.figure(figsize=(12, 6))
    plt.imshow(np.transpose(grid.numpy(), (1, 2, 0)))
    
    # 5. Styling
    plt.title(f"Generated CIFAR-10 Images\nBatch Size: {batch_size} | Steps: {steps} | Inference Time: {time_taken:.4f}s")
    plt.axis("off")
    
    # 6. Save and show
    plt.tight_layout()
    plt.savefig("generated_batch_results.png", dpi=300)
    plt.show()
    
    return time_taken





class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, time_emb_dim):
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_c))
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1)
        
        # SOTA FIX: Dynamic GroupNorm to prevent divisibility errors
        n_groups = 8 if out_c % 8 == 0 else (4 if out_c % 4 == 0 else 1)
        self.norm1 = nn.GroupNorm(n_groups, out_c) # Norm after first conv
        self.norm2 = nn.GroupNorm(n_groups, out_c)
        
        self.res_conv = nn.Conv2d(in_c, out_c, 1) if in_c != out_c else nn.Identity()

    def forward(self, x, t):
        # Time embedding injection
        t_emb = self.mlp(t)[:, :, None, None]
        
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.silu(h + t_emb)
        
        h = self.conv2(h)
        h = self.norm2(h)
        h = F.silu(h)
        
        return h + self.res_conv(x)

class SOTA_UNet(nn.Module):
    def __init__(self, c_in=4, c_out=4, dim=128):
        super().__init__()
        self.time_mlp = nn.Sequential(nn.Linear(1, dim), nn.SiLU(), nn.Linear(dim, dim))
        
        # Encoder (Down)
        self.down1 = ResidualBlock(c_in, dim, dim)      # 8x8
        self.down2 = ResidualBlock(dim, dim * 2, dim)   # 4x4
        self.down3 = ResidualBlock(dim * 2, dim * 4, dim) # 2x2
        
        # Decoder (Up)
        # up3: Cat(upsampled 2x2 @ 512, skip 4x4 @ 256) -> 768 channels
        self.up3 = ResidualBlock(dim * 4 + dim * 2, dim * 2, dim) 
        # up2: Cat(upsampled 4x4 @ 256, skip 8x8 @ 128) -> 384 channels
        self.up2 = ResidualBlock(dim * 2 + dim, dim, dim)
        
        self.final_conv = nn.Conv2d(dim, c_out, 1)
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

    def forward(self, x, t):
        t = self.time_mlp(t.float().view(-1, 1) / 1000)
        
        # Encoder
        h1 = self.down1(x, t)                # [batch, 128, 8, 8]
        h2 = self.down2(self.pool(h1), t)    # [batch, 256, 4, 4]
        h3 = self.down3(self.pool(h2), t)    # [batch, 512, 2, 2]
        
        # Decoder
        # 1. Upsample h3 (2x2 -> 4x4) and concat with h2
        up_h3 = self.upsample(h3)
        x = torch.cat([up_h3, h2], dim=1)    # 512 + 256 = 768
        x = self.up3(x, t)                   # [batch, 256, 4, 4]
        
        # 2. Upsample x (4x4 -> 8x8) and concat with h1
        up_x = self.upsample(x)
        x = torch.cat([up_x, h1], dim=1)     # 256 + 128 = 384
        x = self.up2(x, t)                   # [batch, 128, 8, 8]
        
        return self.final_conv(x)
    
    
# --- 2. THE VAE (SOTA Latent Compression) ---
class SOTA_VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, 4, 3, stride=2, padding=1) # 32x32 -> 8x8
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(4, 128, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 3, 3, padding=1), nn.Tanh()
        )
        
        

# --- 1. SETUP & LOAD MODELS ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# Re-initialize architectures (ensure these match your training script exactly)
vae = SOTA_VAE().to(DEVICE)
student = SOTA_UNet().to(DEVICE)

# Load saved weights
vae.load_state_dict(torch.load("vae_model.pt"))
student.load_state_dict(torch.load("distilled_student.pt"))
vae.eval()
student.eval()

# --- 2. PREPARE REAL DATA FOR FID ---
# FID needs real images to compare against
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
real_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
real_loader = DataLoader(real_dataset, batch_size=100, shuffle=False)

# Metrics (using 2048-dim Inception features is the standard for FID)
fid_metric = FrechetInceptionDistance(feature=2048).to(DEVICE)
is_metric = InceptionScore().to(DEVICE)

# --- 3. MEASURE INFERENCE TIME (32 IMAGES) ---
def measure_inference(batch_size=32, steps=8):
    z = torch.randn(batch_size, 4, 8, 8).to(DEVICE) # Latent noise
    
    start_time = time.time()
    with torch.no_grad():
        # Denoising loop (Student only needs a few steps)
        for i in reversed(range(steps)):
            t = torch.full((batch_size,), i, device=DEVICE)
            v = student(z, t)
            # Simple Euler step for inference
            z = z - (1.0/steps) * v 
        
        # Decode latents to pixels
        imgs = vae.decoder(z)
    
    end_time = time.time()
    elapsed = end_time - start_time
    return imgs, elapsed

# Run timing test
# total_time = plot_generated_images(student, vae, batch_size=32, steps=8)
vae.eval()
x_real, _ = next(iter(real_loader)) 
x_real = x_real[:4].to(DEVICE) # Take 4 real CIFAR images

with torch.no_grad():
    latents = vae.encoder(x_real)
    reconstruction = vae.decoder(latents)

# Plotting
plt.figure(figsize=(8, 4))
for i in range(4):
    plt.subplot(2, 4, i+1); plt.imshow(x_real[i].cpu().permute(1,2,0)*0.5+0.5); plt.title("Original")
    plt.subplot(2, 4, i+5); plt.imshow(reconstruction[i].cpu().permute(1,2,0)*0.5+0.5); plt.title("VAE Rec")
plt.show()


# _, time_taken = measure_inference(32)
# print(f"⏱️ Time for 32 images ({8} steps): {time_taken:.4f} seconds")
# with open("inference_time.txt", "w") as f:
#     f.write(f"Inference Time (32 imgs): {time_taken:.4f}s")

# # --- 4. CALCULATE FID & IS (Large Scale) ---
# print("\n📊 Calculating FID and IS (this takes a moment)...")

# # Add real images to FID
# for real_imgs, _ in real_loader:
#     # Convert to uint8 [0, 255] as expected by torchmetrics FID
#     real_imgs_uint8 = ((real_imgs * 0.5 + 0.5) * 255).to(torch.uint8).to(DEVICE)
#     fid_metric.update(real_imgs_uint8, real=True)

# # Generate fake images and add to FID/IS
# num_gen = 1000 # Use at least 1000 for a decent estimate
# batch_size = 50
# for _ in range(num_gen // batch_size):
#     fake_imgs, _ = measure_inference(batch_size=batch_size)
#     fake_imgs_uint8 = ((fake_imgs * 0.5 + 0.5) * 255).to(torch.uint8).to(DEVICE)
    
#     fid_metric.update(fake_imgs_uint8, real=False)
#     is_metric.update(fake_imgs_uint8)

# # Compute final scores
# fid_score = fid_metric.compute()
# is_mean, is_std = is_metric.compute()

# print(f"✅ FID: {fid_score.item():.2f}")
# print(f"✅ Inception Score: {is_mean.item():.2f} ± {is_std.item():.2f}")