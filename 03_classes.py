import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

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