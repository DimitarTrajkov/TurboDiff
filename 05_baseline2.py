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

# ==========================================
# 1. Architecture Components
# ==========================================
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

class LinearAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = F.elu(q) + 1.0
        k = F.elu(k) + 1.0

        kv = torch.matmul(k.transpose(-2, -1), v) 
        out = torch.matmul(q, kv) 
        
        k_sum = k.sum(dim=-2) 
        denom = torch.matmul(q, k_sum.unsqueeze(-1)).squeeze(-1) + 1e-6
        out = out / denom.unsqueeze(-1)
        
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.proj(out)

class SwiGLU(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(in_features, hidden_features)
        self.w3 = nn.Linear(hidden_features, in_features)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class DiTBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = LinearAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = SwiGLU(dim, dim * 4)
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
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, dim) * 0.02)
        
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
        c = self.t_embedder(t * 1000.0)
        
        for block in self.blocks:
            x = block(x, c)
            
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = self.norm_final(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        x = self.proj_out(x)
        return self.unpatchify(x)

# ==========================================
# 2. Sampling & Evaluation
# ==========================================
@torch.no_grad()
def sample_euler(model, batch_size, device, steps=30):
    model.eval()
    x_t = torch.randn(batch_size, 3, 32, 32, device=device)
    dt = 1.0 / steps
    
    start_time = time.time()
    for i in range(steps):
        t = torch.full((batch_size,), i * dt, device=device)
        v_pred = model(x_t, t)
        x_t = x_t + v_pred * dt 
        
    generation_time = time.time() - start_time
    images = torch.clamp(x_t, -1.0, 1.0)
    model.train()
    return images, generation_time

def evaluate_metrics(model, dataloader, device, num_samples=32, steps=30, epoch=None):
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

    # Note: Passed model should ideally be the EMA model!
    fake_imgs, gen_time = sample_euler(model, num_samples, device, steps=steps)
    fake_imgs_uint8 = ((fake_imgs + 1.0) * 127.5).clamp(0, 255).byte()
    
    fid.update(fake_imgs_uint8, real=False)
    inc.update(fake_imgs_uint8)

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
    
    
def is_score(model, dataloader, device, num_samples=32, steps=30, epoch=None):
    print(f"\n--- Evaluating with {num_samples} samples ---")
    inc = InceptionScore().to(device)

    real_imgs = []
    for batch, _ in dataloader:
        real_imgs.append(batch)
        if sum(b.shape[0] for b in real_imgs) >= num_samples:
            break
    real_imgs = torch.cat(real_imgs, dim=0)[:num_samples].to(device)
    

    # Note: Passed model should ideally be the EMA model!
    fake_imgs, gen_time = sample_euler(model, num_samples, device, steps=steps)
    fake_imgs_uint8 = ((fake_imgs + 1.0) * 127.5).clamp(0, 255).byte()
    
    inc.update(fake_imgs_uint8)

    is_score_mean, is_score_std = inc.compute()
    
    print(f"Generation Time: {gen_time:.4f} seconds")
    print(f"Inception Score: {is_score_mean.item():.4f} ± {is_score_std.item():.4f}")

    grid = make_grid(fake_imgs[:32], nrow=8, normalize=True, value_range=(-1, 1))
    grid_np = grid.permute(1, 2, 0).cpu().numpy()
    
    plt.imshow(grid_np)
    plt.title(f"Generated Samples | IS: {is_score_mean.item():.2f})")
    plt.axis('off')
    plt.savefig(f"generated_samples_{epoch}.png", bbox_inches='tight', dpi=150)
    plt.clf() 
    plt.close()
    return is_score_mean.item()
    
# ==========================================
# 3. Main Training Loop
# ==========================================
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    # torch.backends.cuda.matmul.allow_tf32 = True
    # torch.backends.cudnn.allow_tf32 = True

    BATCH_SIZE = 128
    EPOCHS = 200
    LR = 3e-4

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
    ])
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)

    model = FastDiT(img_size=32, patch_size=4, in_channels=3, dim=256, depth=8, num_heads=8).to(device)    
    
    # optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4, fused=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    
    # 1. EMA (Exponential Moving Average) Setup
    ema_avg = get_ema_multi_avg_fn(0.9999)
    ema_model = AveragedModel(model, multi_avg_fn=ema_avg)
    
    # 2. AMP (Automatic Mixed Precision) Setup
    scaler = torch.amp.GradScaler('cuda')

    # 3. Cosine Scheduler Setup
    total_steps = EPOCHS * len(dataloader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=LR, 
        total_steps=total_steps, 
        pct_start=0.1,  # Warmup for first 10% of training
        anneal_strategy='cos'
    )

    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")
    best_is = 0.0
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for imgs, _ in pbar:
            imgs = imgs.to(device)
            B = imgs.shape[0]

            x_1 = imgs 
            x_0 = torch.randn_like(x_1) 
            t = torch.rand(B, 1, 1, 1, device=device) 
            
            x_t = t * x_1 + (1.0 - t) * x_0
            target_v = x_1 - x_0
            
            optimizer.zero_grad(set_to_none=True)
            
            # AMP Context Manager for speed
            with torch.amp.autocast('cuda'):
                pred_v = model(x_t, t.squeeze())
                loss = F.mse_loss(pred_v, target_v)
            
            # Scaler handles backprop safely with float16
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            scheduler.step()
            
            # Update EMA weights
            ema_model.update_parameters(model)
            
            total_loss += loss.item()
            current_lr = scheduler.get_last_lr()[0]
            pbar.set_postfix({'Loss': f"{loss.item():.4f}", 'LR': f"{current_lr:.2e}"})

        if epoch % 10 == 0:
            # Evaluate using the EMA model for much better results
            curr_is = is_score(ema_model, dataloader, device, num_samples=32, steps=30, epoch=epoch+1)
            if curr_is > best_is:
                best_is = curr_is
                torch.save(ema_model.state_dict(), f"fast_dit_cifar10_epoch{epoch+1}_best.pt")
                    
    # Evaluate using the EMA model for much better results
    evaluate_metrics(ema_model, dataloader, device, num_samples=32, steps=30, epoch=epoch+1)
    torch.save(ema_model.state_dict(), f"fast_dit_cifar10_epoch{epoch+1}.pt")
    
    
# Epoch 1/200: Loss=0.4215, LR=1.38e-05
# --- Evaluating with 32 samples ---
# Generation Time: 0.7579 seconds
# Inception Score: 1.1739 ± 0.0676

# Epoch 2/200: Loss=0.3567, LR=1.90e-05
# Epoch 3/200: Loss=0.3296, LR=2.77e-05
# Epoch 4/200: Loss=0.2968, LR=3.95e-05
# Epoch 5/200: Loss=0.3071, LR=5.42e-05
# Epoch 6/200: Loss=0.2760, LR=7.14e-05
# Epoch 7/200: Loss=0.2774, LR=9.06e-05
# Epoch 8/200: Loss=0.2798, LR=1.12e-04
# Epoch 9/200: Loss=0.2647, LR=1.33e-04
# Epoch 10/200: Loss=0.2559, LR=1.56e-04
# Epoch 11/200: Loss=0.2911, LR=1.79e-04

# --- Evaluating with 32 samples ---
# Generation Time: 0.7966 seconds
# Inception Score: 1.1238 ± 0.0422
# Epoch 12/200: Loss=0.2620, LR=2.01e-04
# Epoch 13/200: Loss=0.2464, LR=2.21e-04
# Epoch 14/200: Loss=0.2460, LR=2.41e-04
# Epoch 15/200: Loss=0.2597, LR=2.58e-04
# Epoch 16/200: Loss=0.2515, LR=2.73e-04
# Epoch 17/200: Loss=0.2432, LR=2.84e-04
# Epoch 18/200: Loss=0.2293, LR=2.93e-04
# Epoch 19/200: Loss=0.2514, LR=2.98e-04
# Epoch 20/200: Loss=0.2372, LR=3.00e-04
# Epoch 21/200: Loss=0.2390, LR=3.00e-04

# --- Evaluating with 32 samples ---
# Generation Time: 0.8600 seconds
# Inception Score: 1.1656 ± 0.0510
# Epoch 22/200: Loss=0.2300, LR=3.00e-04
# Epoch 23/200: Loss=0.2405, LR=3.00e-04
# Epoch 24/200: Loss=0.2016, LR=3.00e-04
# Epoch 25/200: Loss=0.2265, LR=2.99e-04
# Epoch 26/200: Loss=0.2269, LR=2.99e-04
# Epoch 27/200: Loss=0.2147, LR=2.99e-04
# Epoch 28/200: Loss=0.1985, LR=2.99e-04
# Epoch 29/200: Loss=0.1902, LR=2.98e-04
# Epoch 30/200: Loss=0.2028, LR=2.98e-04
# Epoch 31/200: Loss=0.2070, LR=2.97e-04

# --- Evaluating with 32 samples ---
# Generation Time: 0.8198 seconds
# Inception Score: 1.3175 ± 0.1656
# Epoch 32/200: Loss=0.2278, LR=2.97e-04
# Epoch 33/200: Loss=0.2210, LR=2.96e-04
# Epoch 34/200: Loss=0.2167, LR=2.96e-04
# Epoch 35/200: Loss=0.1830, LR=2.95e-04
# Epoch 36/200: Loss=0.1999, LR=2.94e-04
# Epoch 37/200: Loss=0.1945, LR=2.93e-04
# Epoch 38/200: Loss=0.2165, LR=2.93e-04
# Epoch 39/200: Loss=0.1883, LR=2.92e-04
# Epoch 40/200: Loss=0.1832, LR=2.91e-04
# Epoch 41/200: Loss=0.1879, LR=2.90e-04

# --- Evaluating with 32 samples ---
# Generation Time: 0.7859 seconds
# Inception Score: 1.6973 ± 0.1639
# Epoch 42/200: Loss=0.1945, LR=2.89e-04
# Epoch 43/200: Loss=0.1833, LR=2.88e-04
# Epoch 44/200: Loss=0.2100, LR=2.87e-04
# Epoch 45/200: Loss=0.1876, LR=2.86e-04
# Epoch 46/200: Loss=0.1829, LR=2.85e-04
# Epoch 47/200: Loss=0.2069, LR=2.84e-04
# Epoch 48/200: Loss=0.1953, LR=2.82e-04
# Epoch 49/200: Loss=0.2161, LR=2.81e-04
# Epoch 50/200: Loss=0.1864, LR=2.80e-04
# Epoch 51/200: Loss=0.1950, LR=2.79e-04

# --- Evaluating with 32 samples ---
# Generation Time: 0.8118 seconds
# Inception Score: 1.9054 ± 0.2899
# Epoch 52/200: Loss=0.2169, LR=2.77e-04
# Epoch 53/200: Loss=0.1858, LR=2.76e-04
# Epoch 54/200: Loss=0.1856, LR=2.74e-04
# Epoch 55/200: Loss=0.1800, LR=2.73e-04
# Epoch 56/200: Loss=0.1894, LR=2.71e-04
# Epoch 57/200: Loss=0.1755, LR=2.70e-04
# Epoch 58/200: Loss=0.1890, LR=2.68e-04
# Epoch 59/200: Loss=0.1925, LR=2.67e-04
# Epoch 60/200: Loss=0.1873, LR=2.65e-04
# Epoch 61/200: Loss=0.1926, LR=2.63e-04

# --- Evaluating with 32 samples ---
# Generation Time: 0.8284 seconds
# Inception Score: 1.9162 ± 0.2711
# Epoch 62/200: Loss=0.1852, LR=2.61e-04
# Epoch 63/200: Loss=0.1969, LR=2.60e-04
# Epoch 64/200: Loss=0.1923, LR=2.58e-04
# Epoch 65/200: Loss=0.1864, LR=2.56e-04
# Epoch 66/200: Loss=0.1829, LR=2.54e-04
# Epoch 67/200: Loss=0.1865, LR=2.52e-04
# Epoch 68/200: Loss=0.1951, LR=2.50e-04
# Epoch 69/200: Loss=0.1837, LR=2.48e-04
# Epoch 70/200: Loss=0.1932, LR=2.46e-04
# Epoch 71/200: Loss=0.1802, LR=2.44e-04

# --- Evaluating with 32 samples ---
# Generation Time: 0.7841 seconds
# Inception Score: 2.1660 ± 0.3266
# Epoch 72/200: Loss=0.2062, LR=2.42e-04
# Epoch 73/200: Loss=0.1772, LR=2.40e-04
# Epoch 74/200: Loss=0.1962, LR=2.38e-04
# Epoch 75/200: Loss=0.1682, LR=2.36e-04
# Epoch 76/200: Loss=0.1772, LR=2.34e-04
# Epoch 77/200: Loss=0.1832, LR=2.32e-04
# Epoch 78/200: Loss=0.1946, LR=2.29e-04
# Epoch 79/200: Loss=0.1862, LR=2.27e-04
# Epoch 80/200: Loss=0.2005, LR=2.25e-04
# Epoch 81/200: Loss=0.1887, LR=2.23e-04

# --- Evaluating with 32 samples ---
# Generation Time: 0.7471 seconds
# Inception Score: 2.0775 ± 0.4188
# Epoch 82/200: Loss=0.1817, LR=2.20e-04
# Epoch 83/200: Loss=0.1777, LR=2.18e-04
# Epoch 84/200: Loss=0.1782, LR=2.16e-04
# Epoch 85/200: Loss=0.1994, LR=2.13e-04
# Epoch 86/200: Loss=0.1857, LR=2.11e-04
# Epoch 87/200: Loss=0.1737, LR=2.09e-04
# Epoch 88/200: Loss=0.1867, LR=2.06e-04
# Epoch 89/200: Loss=0.1796, LR=2.04e-04
# Epoch 90/200: Loss=0.1667, LR=2.01e-04
# Epoch 91/200: Loss=0.1919, LR=1.99e-04

# --- Evaluating with 32 samples ---
# Generation Time: 0.7942 seconds
# Inception Score: 2.2198 ± 0.4273
# Epoch 92/200: Loss=0.1759, LR=1.96e-04
# Epoch 93/200: Loss=0.1864, LR=1.94e-04
# Epoch 94/200: Loss=0.1739, LR=1.91e-04
# Epoch 95/200: Loss=0.1837, LR=1.89e-04
# Epoch 96/200: Loss=0.1933, LR=1.86e-04
# Epoch 97/200: Loss=0.1906, LR=1.84e-04
# Epoch 98/200: Loss=0.1931, LR=1.81e-04
# Epoch 99/200: Loss=0.1965, LR=1.79e-04
# Epoch 100/200: Loss=0.1771, LR=1.76e-04
# Epoch 101/200: Loss=0.1803, LR=1.73e-04

# --- Evaluating with 32 samples ---
# Generation Time: 0.7820 seconds
# Inception Score: 2.1949 ± 0.3485
# Epoch 102/200: Loss=0.1810, LR=1.71e-04
# Epoch 103/200: Loss=0.1932, LR=1.68e-04
# Epoch 104/200: Loss=0.1909, LR=1.66e-04
# Epoch 105/200: Loss=0.1697, LR=1.63e-04
# Epoch 106/200: Loss=0.1895, LR=1.60e-04
# Epoch 107/200: Loss=0.1998, LR=1.58e-04
# Epoch 108/200: Loss=0.1651, LR=1.55e-04
# Epoch 109/200: Loss=0.1769, LR=1.53e-04
# Epoch 110/200: Loss=0.2005, LR=1.50e-04
# Epoch 111/200: Loss=0.1966, LR=1.47e-04

# --- Evaluating with 32 samples ---
# Generation Time: 0.7888 seconds
# Inception Score: 2.3038 ± 0.4096
# Epoch 112/200: Loss=0.1821, LR=1.45e-04
# Epoch 113/200: Loss=0.1854, LR=1.42e-04
# Epoch 114/200: Loss=0.1824, LR=1.40e-04
# Epoch 115/200: Loss=0.1636, LR=1.37e-04
# Epoch 116/200: Loss=0.1796, LR=1.34e-04
# Epoch 117/200: Loss=0.1911, LR=1.32e-04
# Epoch 118/200: Loss=0.1906, LR=1.29e-04
# Epoch 119/200: Loss=0.1887, LR=1.27e-04
# Epoch 120/200: Loss=0.1822, LR=1.24e-04
# Epoch 121/200: Loss=0.1930, LR=1.21e-04

# --- Evaluating with 32 samples ---
# Generation Time: 0.8841 seconds
# Inception Score: 2.1682 ± 0.4487
# Epoch 122/200: Loss=0.1844, LR=1.19e-04
# Epoch 123/200: Loss=0.1887, LR=1.16e-04
# Epoch 124/200: Loss=0.1991, LR=1.14e-04
# Epoch 125/200: Loss=0.1716, LR=1.11e-04
# Epoch 126/200: Loss=0.1800, LR=1.09e-04
# Epoch 127/200: Loss=0.1781, LR=1.06e-04
# Epoch 128/200: Loss=0.1599, LR=1.04e-04
# Epoch 129/200: Loss=0.1770, LR=1.01e-04
# Epoch 130/200: Loss=0.1874, LR=9.87e-05
# Epoch 131/200: Loss=0.1795, LR=9.62e-05

# --- Evaluating with 32 samples ---
# Generation Time: 0.8059 seconds
# Inception Score: 2.3266 ± 0.3778
# Epoch 132/200: Loss=0.1783, LR=9.38e-05
# Epoch 133/200: Loss=0.1920, LR=9.14e-05
# Epoch 134/200: Loss=0.1936, LR=8.90e-05
# Epoch 135/200: Loss=0.1699, LR=8.66e-05
# Epoch 136/200: Loss=0.1990, LR=8.42e-05
# Epoch 137/200: Loss=0.1790, LR=8.19e-05
# Epoch 138/200: Loss=0.1665, LR=7.96e-05
# Epoch 139/200: Loss=0.1873, LR=7.73e-05
# Epoch 140/200: Loss=0.1886, LR=7.50e-05
# Epoch 141/200: Loss=0.1920, LR=7.27e-05

# --- Evaluating with 32 samples ---
# Generation Time: 0.7594 seconds
# Inception Score: 2.2380 ± 0.2044
# Epoch 142/200: Loss=0.1772, LR=7.05e-05
# Epoch 143/200: Loss=0.1764, LR=6.83e-05
# Epoch 144/200: Loss=0.1684, LR=6.61e-05
# Epoch 145/200: Loss=0.1731, LR=6.40e-05
# Epoch 146/200: Loss=0.1736, LR=6.18e-05
# Epoch 147/200: Loss=0.1711, LR=5.97e-05
# Epoch 148/200: Loss=0.1645, LR=5.76e-05
# Epoch 149/200: Loss=0.1878, LR=5.56e-05
# Epoch 150/200: Loss=0.1828, LR=5.36e-05
# Epoch 151/200: Loss=0.2011, LR=5.16e-05

# --- Evaluating with 32 samples ---
# Generation Time: 0.7979 seconds
# Inception Score: 2.3620 ± 0.1801
# Epoch 152/200: Loss=0.1792, LR=4.96e-05
# Epoch 153/200: Loss=0.2058, LR=4.77e-05
# Epoch 154/200: Loss=0.1832, LR=4.58e-05
# Epoch 155/200: Loss=0.1791, LR=4.39e-05
# Epoch 156/200: Loss=0.1737, LR=4.21e-05
# Epoch 157/200: Loss=0.1787, LR=4.03e-05
# Epoch 158/200: Loss=0.1884, LR=3.85e-05
# Epoch 159/200: Loss=0.1853, LR=3.68e-05
# Epoch 160/200: Loss=0.1675, LR=3.51e-05
# Epoch 161/200: Loss=0.1905, LR=3.34e-05

# --- Evaluating with 32 samples ---
# Generation Time: 0.7572 seconds
# Inception Score: 2.5055 ± 0.2069
# Epoch 162/200: Loss=0.1953, LR=3.18e-05
# Epoch 163/200: Loss=0.1653, LR=3.02e-05
# Epoch 164/200: Loss=0.1841, LR=2.86e-05
# Epoch 165/200: Loss=0.1703, LR=2.71e-05
# Epoch 166/200: Loss=0.1805, LR=2.56e-05
# Epoch 167/200: Loss=0.1824, LR=2.42e-05
# Epoch 168/200: Loss=0.1673, LR=2.28e-05
# Epoch 169/200: Loss=0.1645, LR=2.14e-05
# Epoch 170/200: Loss=0.1874, LR=2.01e-05
# Epoch 171/200: Loss=0.1901, LR=1.88e-05

# --- Evaluating with 32 samples ---
# Generation Time: 0.7860 seconds
# Inception Score: 2.2213 ± 0.2543
# Epoch 172/200: Loss=0.1605, LR=1.76e-05
# Epoch 173/200: Loss=0.1714, LR=1.63e-05
# Epoch 174/200: Loss=0.1755, LR=1.52e-05
# Epoch 175/200: Loss=0.1890, LR=1.41e-05
# Epoch 176/200: Loss=0.1755, LR=1.30e-05
# Epoch 177/200: Loss=0.1720, LR=1.19e-05
# Epoch 178/200: Loss=0.1738, LR=1.09e-05
# Epoch 179/200: Loss=0.1743, LR=9.96e-06
# Epoch 180/200: Loss=0.1816, LR=9.04e-06
# Epoch 181/200: Loss=0.1720, LR=8.17e-06

# --- Evaluating with 32 samples ---
# Generation Time: 0.7941 seconds
# Inception Score: 2.2354 ± 0.2189
# Epoch 182/200: Loss=0.1694, LR=7.34e-06
# Epoch 183/200: Loss=0.2012, LR=6.55e-06
# Epoch 184/200: Loss=0.1691, LR=5.81e-06
# Epoch 185/200: Loss=0.1709, LR=5.11e-06
# Epoch 186/200: Loss=0.1916, LR=4.46e-06
# Epoch 187/200: Loss=0.1768, LR=3.84e-06
# Epoch 188/200: Loss=0.1645, LR=3.28e-06
# Epoch 189/200: Loss=0.1648, LR=2.76e-06
# Epoch 190/200: Loss=0.1689, LR=2.28e-06
# Epoch 191/200: Loss=0.1737, LR=1.85e-06

# --- Evaluating with 32 samples ---
# Generation Time: 0.7730 seconds
# Inception Score: 1.9925 ± 0.1483
# Epoch 192/200: Loss=0.1927, LR=1.46e-06
# Epoch 193/200: Loss=0.1846, LR=1.12e-06
# Epoch 194/200: Loss=0.1679, LR=8.22e-07
# Epoch 195/200: Loss=0.1611, LR=5.71e-07
# Epoch 196/200: Loss=0.1619, LR=3.66e-07
# Epoch 197/200: Loss=0.1661, LR=2.06e-07
# Epoch 198/200: Loss=0.1778, LR=9.23e-08
# Epoch 199/200: Loss=0.1713, LR=2.39e-08
# Epoch 200/200: Loss=0.1812, LR=1.20e-09

# --- Evaluating with 32 samples ---
# Generation Time: 0.5636 seconds
# FID Score: 0.9898
# Inception Score: 2.4535 ± 0.2678


# --- Evaluating with 10000 samples (batch=512) ---
# Generation Time: 765.15s
# FID: 0.1152
# IS: 5.9604 ± 0.1516