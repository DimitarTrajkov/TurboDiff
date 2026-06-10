import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
import math
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
import numpy as np

# ==========================================
# 1. DDPM Cosine Schedule & Min-SNR
# ==========================================
class DDPMSchedule:
    def __init__(self, num_timesteps=1000, s=0.008, device='cuda'):
        self.num_timesteps = num_timesteps
        
        # SOTA 1: Cosine Schedule (Improved DDPM)
        steps = num_timesteps + 1
        x = torch.linspace(0, num_timesteps, steps, dtype=torch.float64)
        alphas_cumprod = torch.cos(((x / num_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = torch.clip(betas, 0.0001, 0.9999).to(torch.float32).to(device)
        
        self.alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Pre-calculate SNR for Min-SNR Weighting
        self.snr = self.alphas_cumprod / (1.0 - self.alphas_cumprod)

# ==========================================
# 2. SOTA Architecture Components
# ==========================================
def get_2d_sincos_pos_embed(embed_dim, grid_size):
    """SOTA 2: Fixed 2D Sine-Cosine Positional Embeddings (from official DiT)"""
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    pos_embed = np.concatenate([emb_h, emb_w], axis=1)
    return torch.from_numpy(pos_embed).float()

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= (embed_dim / 2.)
    omega = 1. / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    return np.concatenate([emb_sin, emb_cos], axis=1)

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    def forward(self, t):
        half_dim = self.frequency_embedding_size // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)
        return self.mlp(emb)

class Attention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj(x)

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.GELU(),
            nn.Linear(hidden_features, in_features)
        )
    def forward(self, x):
        return self.net(x)

class DiTBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = MLP(dim, dim * 4)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x_attn = self.norm1(x) * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        x = x + gate_msa.unsqueeze(1) * self.attn(x_attn)
        x_mlp = self.norm2(x) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_mlp)
        return x

class FastDiT(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3, dim=256, depth=8, num_heads=8):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_patches = (img_size // patch_size) ** 2
        
        self.x_embedder = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)
        self.t_embedder = TimestepEmbedder(dim)
        
        # SOTA 2 Implementation
        pos_embed = get_2d_sincos_pos_embed(dim, img_size // patch_size)
        self.register_buffer("pos_embed", pos_embed.unsqueeze(0))
        
        self.blocks = nn.ModuleList([DiTBlock(dim, num_heads) for _ in range(depth)])
        self.norm_final = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim, bias=True))
        self.proj_out = nn.Linear(dim, patch_size * patch_size * in_channels)

    def unpatchify(self, x):
        B, N, _ = x.shape
        p = self.patch_size
        h = w = int(math.sqrt(N))
        x = x.reshape(B, h, w, self.in_channels, p, p)
        x = torch.einsum('nhwcpq->nchpwq', x)
        return x.reshape(B, self.in_channels, h * p, w * p)

    def forward(self, x, t):
        x = self.x_embedder(x).flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        c = self.t_embedder(t)
        
        for block in self.blocks:
            x = block(x, c)
            
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = self.norm_final(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        x = self.proj_out(x)
        return self.unpatchify(x)

# ==========================================
# 3. Sampling (DDIM with v-prediction)
# ==========================================
@torch.no_grad()
def sample_ddim(model, schedule, batch_size, device, steps=30):
    """DDIM heavily modified to support v-prediction algebra"""
    model.eval()
    x_t = torch.randn(batch_size, 3, 32, 32, device=device)
    
    step_size = schedule.num_timesteps // steps
    timesteps = list(range(0, schedule.num_timesteps, step_size))[::-1]
    
    start_time = time.time()
    for i, t in enumerate(timesteps):
        t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.float32)
        
        # Model now outputs velocity (v)
        pred_v = model(x_t, t_tensor)
        
        alpha_bar_t = schedule.alphas_cumprod[t]
        alpha_bar_t_prev = schedule.alphas_cumprod[t - step_size] if i < steps - 1 else torch.tensor(1.0, device=device)
        
        # SOTA 3: Recover x0 and noise from velocity
        sqrt_alpha = torch.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha = torch.sqrt(1 - alpha_bar_t)
        
        pred_x0 = sqrt_alpha * x_t - sqrt_one_minus_alpha * pred_v
        pred_noise = sqrt_alpha * pred_v + sqrt_one_minus_alpha * x_t
        
        # Point direction to x_{t-1}
        dir_xt = torch.sqrt(1 - alpha_bar_t_prev) * pred_noise
        x_t = torch.sqrt(alpha_bar_t_prev) * pred_x0 + dir_xt
        
    generation_time = time.time() - start_time
    images = torch.clamp(x_t, -1.0, 1.0)
    model.train()
    return images, generation_time

def evaluate_metrics(model, schedule, dataloader, device, num_samples=1_000,batch_size=32, steps=30, epoch=None):
    print(f"\n--- Evaluating with {num_samples} samples ---")
    fid = FrechetInceptionDistance(feature=64).to(device)
    inc = InceptionScore().to(device)

    real_imgs = []
    for batch, _ in dataloader:
        real_imgs.append(batch)
        if sum(b.shape[0] for b in real_imgs) >= num_samples:
            break
    real_imgs = torch.cat(real_imgs, dim=0)[:num_samples].to(device)
    real_imgs_uint8 = ((real_imgs + 1.0) * 127.5).clamp(0, 255).byte()
    fid.update(real_imgs_uint8, real=True)
    
    generated = 0
    start_time = time.time()
    while generated < num_samples:
        current_bs = min(batch_size, num_samples - generated)

        fake_imgs, _ = sample_ddim(model, schedule, num_samples, device, steps=steps)
        fake_imgs_uint8 = ((fake_imgs + 1.0) * 127.5).clamp(0, 255).byte()
        
        fid.update(fake_imgs_uint8, real=False)
        inc.update(fake_imgs_uint8)
        
        generated += current_bs

    gen_time = time.time() - start_time

    fid_score = fid.compute().item()
    is_score_mean, is_score_std = inc.compute()
    
    print(f"Generation Time: {gen_time:.4f} seconds")
    print(f"FID Score: {fid_score:.4f}")
    print(f"Inception Score: {is_score_mean.item():.4f} ± {is_score_std.item():.4f}")

    grid = make_grid(fake_imgs[:32], nrow=8, normalize=True, value_range=(-1, 1))
    grid_np = grid.permute(1, 2, 0).cpu().numpy()
    
    plt.imshow(grid_np)
    plt.title(f"Generated Samples (FID: {fid_score:.2f} | IS: {is_score_mean.item():.2f})")
    plt.axis('off')
    plt.savefig(f"generated_samples_{epoch}.png", bbox_inches='tight', dpi=150)
    plt.clf() 
    plt.close()
    
def is_score(model, schedule, dataloader, device, num_samples=32, steps=30, epoch=None):
    inc = InceptionScore().to(device)
    fake_imgs, gen_time = sample_ddim(model, schedule, num_samples, device, steps=steps)
    fake_imgs_uint8 = ((fake_imgs + 1.0) * 127.5).clamp(0, 255).byte()
    inc.update(fake_imgs_uint8)
    is_score_mean, is_score_std = inc.compute()
    print(f"IS: {is_score_mean.item():.4f} ± {is_score_std.item():.4f} took {gen_time:.4f} seconds")
    
    grid = make_grid(fake_imgs[:32], nrow=8, normalize=True, value_range=(-1, 1))
    grid_np = grid.permute(1, 2, 0).cpu().numpy()
    plt.imshow(grid_np)
    plt.title(f"Generated Samples (IS: {is_score_mean.item():.2f})")
    plt.axis('off')
    plt.savefig(f"generated_samples_{epoch}.png", bbox_inches='tight', dpi=150)
    plt.clf() 
    plt.close()
    
    
    return is_score_mean.item()
    
# ==========================================
# 4. Main Training Loop
# ==========================================
if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    BATCH_SIZE = 128
    EPOCHS = 200
    LR = 3e-4

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
    ])
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)

    model = FastDiT(img_size=32, patch_size=4, in_channels=3, dim=256, depth=8, num_heads=8).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4, fused=True)
    schedule = DDPMSchedule(num_timesteps=1000, device=device)

    ema_avg = get_ema_multi_avg_fn(0.9999)
    ema_model = AveragedModel(model, multi_avg_fn=ema_avg)
    scaler = torch.amp.GradScaler('cuda')

    total_steps = EPOCHS * len(dataloader)
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR, total_steps=total_steps, pct_start=0.1, anneal_strategy='cos'
    )

    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")
    best_is = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for imgs, _ in pbar:
            imgs = imgs.to(device)
            B = imgs.shape[0]

            t = torch.randint(0, schedule.num_timesteps, (B,), device=device)
            noise = torch.randn_like(imgs)
            
            sqrt_alpha_bar = schedule.sqrt_alphas_cumprod[t].view(B, 1, 1, 1)
            sqrt_one_minus_alpha_bar = schedule.sqrt_one_minus_alphas_cumprod[t].view(B, 1, 1, 1)
            
            x_t = sqrt_alpha_bar * imgs + sqrt_one_minus_alpha_bar * noise
            
            # SOTA 3: Target is now Velocity (v), not Noise
            target_v = sqrt_alpha_bar * noise - sqrt_one_minus_alpha_bar * imgs 
            
            # SOTA 4: Min-SNR Loss Weighting (Gamma = 5.0)
            snr_t = schedule.snr[t].view(B, 1, 1, 1)
            loss_weight = torch.clamp(snr_t, max=5.0) / (snr_t + 1.0)

            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast('cuda'):
                pred_v = model(x_t, t.float())
                # Apply Min-SNR weights to MSE
                loss = torch.mean(loss_weight * (pred_v - target_v) ** 2)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            lr_scheduler.step()
            ema_model.update_parameters(model)
            
            current_lr = lr_scheduler.get_last_lr()[0]
            pbar.set_postfix({'Loss': f"{loss.item():.4f}", 'LR': f"{current_lr:.2e}"})

        if epoch % 10 == 0:
            curr_is = is_score(ema_model, schedule, dataloader, device, num_samples=32, steps=30, epoch=epoch+1)
            if curr_is > best_is:
                best_is = curr_is
                torch.save(ema_model.state_dict(), f"ddpm_sota_dit_best.pt")
                
    # torch.save(ema_model.state_dict(), f"ddpm_sota_dit_final.pt")
    # evaluate_metrics(ema_model, schedule, dataloader, device, num_samples=50_000,batch_size=512, steps=30, epoch="final_eval_08")
    

# Model Parameters: 9.76 M
# Epoch 1/200: Loss=0.0707, LR=1.38e-05
# IS: 1.1365 ± 0.0457 took 0.3979 seconds
# Epoch 2/200: Loss=0.0546, LR=1.90e-05
# Epoch 3/200: Loss=0.0498, LR=2.77e-05
# Epoch 4/200: Loss=0.0459, LR=3.95e-05
# Epoch 5/200: Loss=0.0397, LR=5.42e-05
# Epoch 6/200: Loss=0.0377, LR=7.14e-05
# Epoch 7/200: Loss=0.0365, LR=9.06e-05
# Epoch 8/200: Loss=0.0371, LR=1.12e-04
# Epoch 9/200: Loss=0.0333, LR=1.33e-04
# Epoch 10/200: Loss=0.0308, LR=1.56e-04
# Epoch 11/200: Loss=0.0346, LR=1.79e-04
# IS: 1.1505 ± 0.0549 took 0.3246 seconds
# Epoch 12/200: Loss=0.0307, LR=2.01e-04
# Epoch 13/200: Loss=0.0281, LR=2.21e-04
# Epoch 14/200: Loss=0.0299, LR=2.41e-04
# Epoch 15/200: Loss=0.0292, LR=2.58e-04
# Epoch 16/200: Loss=0.0266, LR=2.73e-04
# Epoch 17/200: Loss=0.0284, LR=2.84e-04
# Epoch 18/200: Loss=0.0279, LR=2.93e-04
# Epoch 19/200: Loss=0.0307, LR=2.98e-04
# Epoch 20/200: Loss=0.0284, LR=3.00e-04
# Epoch 21/200: Loss=0.0265, LR=3.00e-04
# IS: 1.1815 ± 0.0818 took 0.3322 seconds
# Epoch 22/200: Loss=0.0260, LR=3.00e-04
# Epoch 23/200: Loss=0.0290, LR=3.00e-04
# Epoch 24/200: Loss=0.0301, LR=3.00e-04
# Epoch 25/200: Loss=0.0263, LR=2.99e-04
# Epoch 26/200: Loss=0.0266, LR=2.99e-04
# Epoch 27/200: Loss=0.0291, LR=2.99e-04
# Epoch 28/200: Loss=0.0289, LR=2.99e-04
# Epoch 29/200: Loss=0.0265, LR=2.98e-04
# Epoch 30/200: Loss=0.0268, LR=2.98e-04
# Epoch 31/200: Loss=0.0274, LR=2.97e-04
# IS: 1.1825 ± 0.0952 took 0.3664 seconds
# Epoch 32/200: Loss=0.0271, LR=2.97e-04
# Epoch 33/200: Loss=0.0233, LR=2.96e-04
# Epoch 34/200: Loss=0.0245, LR=2.96e-04
# Epoch 35/200: Loss=0.0279, LR=2.95e-04
# Epoch 36/200: Loss=0.0244, LR=2.94e-04
# Epoch 37/200: Loss=0.0248, LR=2.93e-04
# Epoch 38/200: Loss=0.0269, LR=2.93e-04
# Epoch 39/200: Loss=0.0255, LR=2.92e-04
# Epoch 40/200: Loss=0.0229, LR=2.91e-04
# Epoch 41/200: Loss=0.0284, LR=2.90e-04
# IS: 1.2692 ± 0.0657 took 0.3578 seconds
# Epoch 42/200: Loss=0.0230, LR=2.89e-04
# Epoch 43/200: Loss=0.0305, LR=2.88e-04
# Epoch 44/200: Loss=0.0237, LR=2.87e-04
# Epoch 45/200: Loss=0.0264, LR=2.86e-04
# Epoch 46/200: Loss=0.0282, LR=2.85e-04
# Epoch 47/200: Loss=0.0279, LR=2.84e-04
# Epoch 48/200: Loss=0.0265, LR=2.82e-04
# Epoch 49/200: Loss=0.0287, LR=2.81e-04
# Epoch 50/200: Loss=0.0256, LR=2.80e-04
# Epoch 51/200: Loss=0.0234, LR=2.79e-04
# IS: 1.2773 ± 0.1445 took 0.3718 seconds
# Epoch 52/200: Loss=0.0280, LR=2.77e-04
# Epoch 53/200: Loss=0.0217, LR=2.76e-04
# Epoch 54/200: Loss=0.0258, LR=2.74e-04
# Epoch 55/200: Loss=0.0247, LR=2.73e-04
# Epoch 56/200: Loss=0.0254, LR=2.71e-04
# Epoch 57/200: Loss=0.0262, LR=2.70e-04
# Epoch 58/200: Loss=0.0262, LR=2.68e-04
# Epoch 59/200: Loss=0.0207, LR=2.67e-04
# Epoch 60/200: Loss=0.0256, LR=2.65e-04
# Epoch 61/200: Loss=0.0239, LR=2.63e-04
# IS: 1.5119 ± 0.2126 took 0.3739 seconds
# Epoch 62/200: Loss=0.0234, LR=2.61e-04
# Epoch 63/200: Loss=0.0277, LR=2.60e-04
# Epoch 64/200: Loss=0.0290, LR=2.58e-04
# Epoch 65/200: Loss=0.0296, LR=2.56e-04
# Epoch 66/200: Loss=0.0246, LR=2.54e-04
# Epoch 67/200: Loss=0.0218, LR=2.52e-04
# Epoch 68/200: Loss=0.0247, LR=2.50e-04
# Epoch 69/200: Loss=0.0246, LR=2.48e-04
# Epoch 70/200: Loss=0.0254, LR=2.46e-04
# Epoch 71/200: Loss=0.0226, LR=2.44e-04
# IS: 1.5954 ± 0.2084 took 0.3377 seconds
# Epoch 72/200: Loss=0.0273, LR=2.42e-04
# Epoch 73/200: Loss=0.0268, LR=2.40e-04
# Epoch 74/200: Loss=0.0260, LR=2.38e-04
# Epoch 75/200: Loss=0.0238, LR=2.36e-04
# Epoch 76/200: Loss=0.0253, LR=2.34e-04
# Epoch 77/200: Loss=0.0232, LR=2.32e-04
# Epoch 78/200: Loss=0.0300, LR=2.29e-04
# Epoch 79/200: Loss=0.0271, LR=2.27e-04
# Epoch 80/200: Loss=0.0260, LR=2.25e-04
# Epoch 81/200: Loss=0.0275, LR=2.23e-04
# IS: 1.8253 ± 0.2612 took 0.3420 seconds
# Epoch 82/200: Loss=0.0278, LR=2.20e-04
# Epoch 83/200: Loss=0.0249, LR=2.18e-04
# Epoch 84/200: Loss=0.0245, LR=2.16e-04
# Epoch 85/200: Loss=0.0236, LR=2.13e-04
# Epoch 86/200: Loss=0.0231, LR=2.11e-04
# Epoch 87/200: Loss=0.0251, LR=2.09e-04
# Epoch 88/200: Loss=0.0275, LR=2.06e-04
# Epoch 89/200: Loss=0.0240, LR=2.04e-04
# Epoch 90/200: Loss=0.0247, LR=2.01e-04
# Epoch 91/200: Loss=0.0282, LR=1.99e-04
# IS: 1.8338 ± 0.1701 took 0.3560 seconds
# Epoch 92/200: Loss=0.0223, LR=1.96e-04
# Epoch 93/200: Loss=0.0247, LR=1.94e-04
# Epoch 94/200: Loss=0.0283, LR=1.91e-04
# Epoch 95/200: Loss=0.0266, LR=1.89e-04
# Epoch 96/200: Loss=0.0221, LR=1.86e-04
# Epoch 97/200: Loss=0.0244, LR=1.84e-04
# Epoch 98/200: Loss=0.0299, LR=1.81e-04
# Epoch 99/200: Loss=0.0272, LR=1.79e-04
# Epoch 100/200: Loss=0.0275, LR=1.76e-04
# Epoch 101/200: Loss=0.0219, LR=1.73e-04
# IS: 2.1628 ± 0.4375 took 0.3465 seconds
# Epoch 102/200: Loss=0.0304, LR=1.71e-04
# Epoch 103/200: Loss=0.0278, LR=1.68e-04
# Epoch 104/200: Loss=0.0239, LR=1.66e-04
# Epoch 105/200: Loss=0.0226, LR=1.63e-04
# Epoch 106/200: Loss=0.0225, LR=1.60e-04
# Epoch 107/200: Loss=0.0258, LR=1.58e-04
# Epoch 108/200: Loss=0.0239, LR=1.55e-04
# Epoch 109/200: Loss=0.0251, LR=1.53e-04
# Epoch 110/200: Loss=0.0306, LR=1.50e-04
# Epoch 111/200: Loss=0.0264, LR=1.47e-04
# IS: 2.2060 ± 0.3645 took 0.3476 seconds
# Epoch 112/200: Loss=0.0245, LR=1.45e-04
# Epoch 113/200: Loss=0.0278, LR=1.42e-04
# Epoch 114/200: Loss=0.0257, LR=1.40e-04
# Epoch 115/200: Loss=0.0263, LR=1.37e-04
# Epoch 116/200: Loss=0.0246, LR=1.34e-04
# Epoch 117/200: Loss=0.0249, LR=1.32e-04
# Epoch 118/200: Loss=0.0233, LR=1.29e-04
# Epoch 119/200: Loss=0.0270, LR=1.27e-04
# Epoch 120/200: Loss=0.0265, LR=1.24e-04
# Epoch 121/200: Loss=0.0276, LR=1.21e-04
# IS: 2.1611 ± 0.3151 took 0.3099 seconds
# Epoch 122/200: Loss=0.0272, LR=1.19e-04
# Epoch 123/200: Loss=0.0209, LR=1.16e-04
# Epoch 124/200: Loss=0.0267, LR=1.14e-04
# Epoch 125/200: Loss=0.0273, LR=1.11e-04
# Epoch 126/200: Loss=0.0254, LR=1.09e-04
# Epoch 127/200: Loss=0.0238, LR=1.06e-04
# Epoch 128/200: Loss=0.0233, LR=1.04e-04
# Epoch 129/200: Loss=0.0247, LR=1.01e-04
# Epoch 130/200: Loss=0.0252, LR=9.87e-05
# Epoch 131/200: Loss=0.0289, LR=9.62e-05
# IS: 2.2805 ± 0.4001 took 0.3519 seconds
# Epoch 132/200: Loss=0.0236, LR=9.38e-05
# Epoch 133/200: Loss=0.0248, LR=9.14e-05
# Epoch 134/200: Loss=0.0295, LR=8.90e-05
# Epoch 135/200: Loss=0.0224, LR=8.66e-05
# Epoch 136/200: Loss=0.0249, LR=8.42e-05
# Epoch 137/200: Loss=0.0249, LR=8.19e-05
# Epoch 138/200: Loss=0.0235, LR=7.96e-05
# Epoch 139/200: Loss=0.0270, LR=7.73e-05
# Epoch 140/200: Loss=0.0277, LR=7.50e-05
# Epoch 141/200: Loss=0.0253, LR=7.27e-05
# IS: 2.3976 ± 0.4585 took 0.3283 seconds
# Epoch 142/200: Loss=0.0246, LR=7.05e-05
# Epoch 143/200: Loss=0.0258, LR=6.83e-05
# Epoch 144/200: Loss=0.0229, LR=6.61e-05
# Epoch 145/200: Loss=0.0273, LR=6.40e-05
# Epoch 146/200: Loss=0.0239, LR=6.18e-05
# Epoch 147/200: Loss=0.0244, LR=5.97e-05
# Epoch 148/200: Loss=0.0236, LR=5.76e-05
# Epoch 149/200: Loss=0.0240, LR=5.56e-05
# Epoch 150/200: Loss=0.0248, LR=5.36e-05
# Epoch 151/200: Loss=0.0262, LR=5.16e-05
# IS: 2.4614 ± 0.2570 took 0.3849 seconds
# Epoch 152/200: Loss=0.0277, LR=4.96e-05
# Epoch 153/200: Loss=0.0254, LR=4.77e-05
# Epoch 154/200: Loss=0.0268, LR=4.58e-05
# Epoch 155/200: Loss=0.0257, LR=4.39e-05
# Epoch 156/200: Loss=0.0279, LR=4.21e-05
# Epoch 157/200: Loss=0.0223, LR=4.03e-05
# Epoch 158/200: Loss=0.0257, LR=3.85e-05
# Epoch 159/200: Loss=0.0258, LR=3.68e-05
# Epoch 160/200: Loss=0.0296, LR=3.51e-05
# Epoch 161/200: Loss=0.0240, LR=3.34e-05
# IS: 2.3953 ± 0.3469 took 0.3793 seconds
# Epoch 162/200: Loss=0.0229, LR=3.18e-05
# Epoch 163/200: Loss=0.0258, LR=3.02e-05
# Epoch 164/200: Loss=0.0249, LR=2.86e-05
# Epoch 165/200: Loss=0.0275, LR=2.71e-05
# Epoch 166/200: Loss=0.0231, LR=2.56e-05
# Epoch 167/200: Loss=0.0265, LR=2.42e-05
# Epoch 168/200: Loss=0.0221, LR=2.28e-05
# Epoch 169/200: Loss=0.0234, LR=2.14e-05
# Epoch 170/200: Loss=0.0250, LR=2.01e-05
# Epoch 171/200: Loss=0.0283, LR=1.88e-05
# IS: 2.2796 ± 0.2119 took 0.3575 seconds
# Epoch 172/200: Loss=0.0255, LR=1.76e-05
# Epoch 173/200: Loss=0.0254, LR=1.63e-05
# Epoch 174/200: Loss=0.0296, LR=1.52e-05
# Epoch 175/200: Loss=0.0238, LR=1.41e-05
# Epoch 176/200: Loss=0.0264, LR=1.30e-05
# Epoch 177/200: Loss=0.0272, LR=1.19e-05
# Epoch 178/200: Loss=0.0277, LR=1.09e-05
# Epoch 179/200: Loss=0.0247, LR=9.96e-06
# Epoch 180/200: Loss=0.0229, LR=9.04e-06
# Epoch 181/200: Loss=0.0260, LR=8.17e-06
# IS: 2.5941 ± 0.4655 took 0.3374 seconds
# Epoch 182/200: Loss=0.0251, LR=7.34e-06
# Epoch 183/200: Loss=0.0241, LR=6.55e-06
# Epoch 184/200: Loss=0.0265, LR=5.81e-06
# Epoch 185/200: Loss=0.0268, LR=5.11e-06
# Epoch 186/200: Loss=0.0247, LR=4.46e-06
# Epoch 187/200: Loss=0.0275, LR=3.84e-06
# Epoch 188/200: Loss=0.0257, LR=3.28e-06
# Epoch 189/200: Loss=0.0263, LR=2.76e-06
# Epoch 190/200: Loss=0.0253, LR=2.28e-06
# Epoch 191/200: Loss=0.0278, LR=1.85e-06
# IS: 2.3227 ± 0.2831 took 0.3975 seconds
# Epoch 192/200: Loss=0.0268, LR=1.46e-06
# Epoch 193/200: Loss=0.0224, LR=1.12e-06
# Epoch 194/200: Loss=0.0257, LR=8.22e-07
# Epoch 195/200: Loss=0.0265, LR=5.71e-07
# Epoch 196/200: Loss=0.0276, LR=3.66e-07
# Epoch 197/200: Loss=0.0215, LR=2.06e-07
# Epoch 198/200: Loss=0.0240, LR=9.23e-08
# Epoch 199/200: Loss=0.0262, LR=2.39e-08
# Epoch 200/200: Loss=0.0254, LR=1.20e-09