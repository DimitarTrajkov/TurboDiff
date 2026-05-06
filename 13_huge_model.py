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
        self.snr = self.alphas_cumprod / (1.0 - self.alphas_cumprod)

# ==========================================
# 2. Pure DiT Architecture (No Bottlenecks)
# ==========================================
def get_2d_sincos_pos_embed(embed_dim, grid_size):
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

class PatchEmbed(nn.Module):
    def __init__(self, in_channels=3, embed_dim=768, patch_size=2):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x)

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
        # Flash Attention triggered automatically here
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
        # AdaLN-Zero Initialization
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x_attn = self.norm1(x) * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        x = x + gate_msa.unsqueeze(1) * self.attn(x_attn)
        x_mlp = self.norm2(x) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_mlp)
        return x

class HeavyDiT(nn.Module):
    def __init__(self, img_size=32, patch_size=2, in_channels=3, dim=768, depth=12, num_heads=12):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        
        self.x_embedder = PatchEmbed(in_channels=in_channels, embed_dim=dim, patch_size=patch_size)
        self.t_embedder = TimestepEmbedder(dim)
        
        grid_size = img_size // patch_size
        pos_embed = get_2d_sincos_pos_embed(dim, grid_size)
        self.register_buffer("pos_embed", pos_embed.unsqueeze(0))
        
        self.blocks = nn.ModuleList([
            DiTBlock(dim, num_heads) for _ in range(depth)
        ])

        self.norm_final = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim, bias=True))
        self.proj_out = nn.Linear(dim, patch_size * patch_size * in_channels)

        # Final AdaLN-Zero Initialization
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

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
def sample_ddim(model, schedule, batch_size, device, steps=50):
    # Increased default steps to 50 for higher quality generation
    model.eval()
    x_t = torch.randn(batch_size, 3, 32, 32, device=device)
    step_size = schedule.num_timesteps // steps
    timesteps = list(range(0, schedule.num_timesteps, step_size))[::-1]
    
    start_time = time.time()
    for i, t in enumerate(timesteps):
        t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.float32)
        pred_v = model(x_t, t_tensor)
        
        alpha_bar_t = schedule.alphas_cumprod[t]
        alpha_bar_t_prev = schedule.alphas_cumprod[t - step_size] if i < steps - 1 else torch.tensor(1.0, device=device)
        
        sqrt_alpha = torch.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha = torch.sqrt(1 - alpha_bar_t)
        
        pred_x0 = sqrt_alpha * x_t - sqrt_one_minus_alpha * pred_v
        pred_noise = sqrt_alpha * pred_v + sqrt_one_minus_alpha * x_t
        
        dir_xt = torch.sqrt(1 - alpha_bar_t_prev) * pred_noise
        x_t = torch.sqrt(alpha_bar_t_prev) * pred_x0 + dir_xt
        
    generation_time = time.time() - start_time
    images = torch.clamp(x_t, -1.0, 1.0)
    model.train()
    return images, generation_time

@torch.no_grad()
def is_score(model, schedule, device, num_samples=2048, eval_batch_size=128, steps=50, epoch=None):
    inc = InceptionScore().to(device)
    generated = 0
    
    save_one_batch = False
    while generated < num_samples:
        current_batch_size = min(eval_batch_size, num_samples - generated)
        fake_imgs, _ = sample_ddim(model, schedule, current_batch_size, device, steps=steps)
        
        fake_imgs_uint8 = ((fake_imgs + 1.0) * 127.5).clamp(0, 255).byte()
        inc.update(fake_imgs_uint8)
        generated += current_batch_size
        
        if not save_one_batch:
            grid = make_grid(fake_imgs[:32], nrow=8, normalize=True, value_range=(-1, 1))
            grid_np = grid.permute(1, 2, 0).cpu().numpy()
            
            plt.imshow(grid_np)
            plt.title(f"Generated Samples")
            plt.axis('off')
            
            epoch_label = f"epoch_{epoch}" if epoch else "final"
            plt.savefig(f"generated_samples_{epoch_label}.png", bbox_inches='tight', dpi=150)
            plt.clf() 
            plt.close()
            save_one_batch = True
            
        del fake_imgs
        del fake_imgs_uint8

    is_score_mean, is_score_std = inc.compute()
    score = is_score_mean.item()
    
    del inc
    torch.cuda.empty_cache()
    return score

# ==========================================
# 4. Main Training Loop
# ==========================================
if __name__ == '__main__':
    # Maximize GPU efficiency
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Dropped batch size slightly to fit the massive 768-dim model in VRAM
    BATCH_SIZE = 64  
    EPOCHS = 500 # Unconditional generation requires massive epoch counts
    LR = 2e-4 # Slightly lower LR for the larger model stability

    # Standard augmentations
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
    ])
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    
    # Maximize CPU-to-GPU data transfer
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)

    # The Heavyweight Model: Dim 768, Depth 12, Patch 2
    model = HeavyDiT(img_size=32, patch_size=2, in_channels=3, dim=768, depth=12, num_heads=12).to(device)
    
    # CRITICAL SPEEDUP: Compile the model to fuse operations
    print("Compiling model for maximum throughput...")
    # model = torch.compile(model)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4, fused=True)
    schedule = DDPMSchedule(num_timesteps=1000, device=device)

    ema_avg = get_ema_multi_avg_fn(0.9999)
    ema_model = AveragedModel(model, multi_avg_fn=ema_avg)
    scaler = torch.amp.GradScaler('cuda')

    total_steps = EPOCHS * len(dataloader)
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR, total_steps=total_steps, pct_start=0.05, anneal_strategy='cos'
    )

    # This should read ~85 Million Parameters now
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
            target_v = sqrt_alpha_bar * noise - sqrt_one_minus_alpha_bar * imgs 
            
            # Min-SNR Weighting Strategy
            snr_t = schedule.snr[t].view(B, 1, 1, 1)
            loss_weight = torch.clamp(snr_t, max=5.0) / (snr_t + 1.0)

            optimizer.zero_grad(set_to_none=True)
            
            # Autocast for Tensor Cores
            with torch.amp.autocast('cuda'):
                pred_v = model(x_t, t.float())
                loss = torch.mean(loss_weight * (pred_v - target_v) ** 2)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            lr_scheduler.step()
            ema_model.update_parameters(model)
            
            current_lr = lr_scheduler.get_last_lr()[0]
            pbar.set_postfix({'Loss': f"{loss.item():.4f}", 'LR': f"{current_lr:.2e}"})

        # Evaluate every 20 epochs to save time, as training will be longer
        if (epoch + 1) % 20 == 0:
            curr_is = is_score(ema_model, schedule, device, num_samples=1024, eval_batch_size=128, steps=50, epoch=epoch+1)
            print(f"\nEpoch {epoch+1} Mini-IS: {curr_is:.2f}")
            if curr_is > best_is:
                best_is = curr_is
                torch.save(ema_model.state_dict(), "heavy_dit_best.pt")
                print(f"New Best IS Model Saved! Score: {best_is:.2f}")

    torch.save(ema_model.state_dict(), "heavy_dit_final.pt")