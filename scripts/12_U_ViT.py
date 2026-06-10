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
# 2. SOTA Architecture Components (U-ViT)
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
    def __init__(self, in_channels=3, embed_dim=384, patch_size=2):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x)

class PatchMerge(nn.Module):
    """Shrinks sequence length by 4x (groups 2x2 tokens into 1)"""
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(dim * 4, dim)

    def forward(self, x):
        B, N, C = x.shape
        h = w = int(math.sqrt(N))
        x = x.view(B, h // 2, 2, w // 2, 2, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, (h // 2) * (w // 2), C * 4)
        return self.proj(x)

class PatchExpand(nn.Module):
    """Expands sequence length by 4x (splits 1 token into 2x2)"""
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(dim, dim * 4)

    def forward(self, x):
        B, N, C = x.shape
        h = w = int(math.sqrt(N))
        x = self.proj(x) 
        x = x.view(B, h, w, 2, 2, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, h * 2 * w * 2, C)
        return x

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

class FastUViT(nn.Module):
    def __init__(self, img_size=32, patch_size=2, in_channels=3, dim=384, inner_depth=10, num_heads=12):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        
        self.x_embedder = PatchEmbed(in_channels=in_channels, embed_dim=dim, patch_size=patch_size)
        self.t_embedder = TimestepEmbedder(dim)
        
        # Dual Positional Embeddings
        high_res_grid = img_size // patch_size       # 16x16
        low_res_grid = high_res_grid // 2            # 8x8
        
        pos_embed_high = get_2d_sincos_pos_embed(dim, high_res_grid)
        pos_embed_low = get_2d_sincos_pos_embed(dim, low_res_grid)
        self.register_buffer("pos_embed_high", pos_embed_high.unsqueeze(0))
        self.register_buffer("pos_embed_low", pos_embed_low.unsqueeze(0))
        
        # Hierarchical Network Blocks
        self.in_block = DiTBlock(dim, num_heads)
        self.merge = PatchMerge(dim)
        
        self.mid_blocks = nn.ModuleList([DiTBlock(dim, num_heads) for _ in range(inner_depth)])
        
        self.expand = PatchExpand(dim)
        self.skip_proj = nn.Linear(dim * 2, dim)
        self.out_block = DiTBlock(dim, num_heads)

        # Output layers
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
        # 1. High Resolution Processing
        x = self.x_embedder(x).flatten(2).transpose(1, 2)
        x = x + self.pos_embed_high
        c = self.t_embedder(t)
        
        x = self.in_block(x, c)
        skip = x  # Save for later
        
        # 2. Downsample to Low Resolution
        x = self.merge(x)
        x = x + self.pos_embed_low
        
        for block in self.mid_blocks:
            x = block(x, c)
            
        # 3. Upsample and Merge High Res Details
        x = self.expand(x)
        x = torch.cat([x, skip], dim=-1)
        x = self.skip_proj(x)
        
        x = self.out_block(x, c)
        
        # 4. Final Projection
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = self.norm_final(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        x = self.proj_out(x)
        return self.unpatchify(x)

# ==========================================
# 3. Sampling (DDIM with v-prediction)
# ==========================================
@torch.no_grad()
def sample_ddim(model, schedule, batch_size, device, steps=30):
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
def evaluate_metrics(model, schedule, dataloader, device, num_samples=50000, eval_batch_size=256, steps=30, epoch=None):
    print(f"\n--- Evaluating with {num_samples} samples ---")
    fid = FrechetInceptionDistance(feature=2048).to(device)
    inc = InceptionScore().to(device)

    # STEP 1: Process Real Images in Chunks
    real_processed = 0
    couter = num_samples// eval_batch_size + 1
    for batch, _ in dataloader:
        if real_processed >= num_samples:
            break
        
        current_batch_size = batch.shape[0]
        if real_processed + current_batch_size > num_samples:
            batch = batch[:num_samples - real_processed]
        
        batch = batch.to(device)
        real_imgs_uint8 = ((batch + 1.0) * 127.5).clamp(0, 255).byte()
        
        fid.update(real_imgs_uint8, real=True)
        real_processed += batch.shape[0]
        couter -=1
        print(f"Processed {real_processed}/{num_samples} real images... ({couter} batches left)", end='\r')
        del batch
        del real_imgs_uint8

    # STEP 2: Process Fake Images in Chunks
    fake_processed = 0
    total_gen_time = 0.0
    fake_imgs_for_grid = []
    couter = num_samples// eval_batch_size + 1
    while fake_processed < num_samples:
        current_batch_size = min(eval_batch_size, num_samples - fake_processed)
        
        fake_imgs, gen_time = sample_ddim(model, schedule, current_batch_size, device, steps=steps)
        total_gen_time += gen_time
        
        fake_imgs_uint8 = ((fake_imgs + 1.0) * 127.5).clamp(0, 255).byte()
        
        fid.update(fake_imgs_uint8, real=False)
        inc.update(fake_imgs_uint8)
        
        if len(fake_imgs_for_grid) < 32:
            needed = 32 - len(fake_imgs_for_grid)
            fake_imgs_for_grid.append(fake_imgs[:needed].cpu())
            
        fake_processed += current_batch_size
        couter -=1
        print(f"Processed {fake_processed}/{num_samples} fake images... ({couter} batches left)", end='\r')
        del fake_imgs
        del fake_imgs_uint8

    # STEP 3: Compute Final Scores and Plot
    fid_score = fid.compute().item()
    is_score_mean, is_score_std = inc.compute()
    
    print(f"Total Generation Time: {total_gen_time:.4f} seconds")
    print(f"FID Score: {fid_score:.4f}")
    print(f"Inception Score: {is_score_mean.item():.4f} ± {is_score_std.item():.4f}")

    grid_imgs = torch.cat(fake_imgs_for_grid, dim=0)
    grid = make_grid(grid_imgs, nrow=8, normalize=True, value_range=(-1, 1))
    grid_np = grid.permute(1, 2, 0).numpy()
    
    plt.imshow(grid_np)
    plt.title(f"Generated Samples (FID: {fid_score:.2f} | IS: {is_score_mean.item():.2f})")
    plt.axis('off')
    
    epoch_label = f"epoch_{epoch}" if epoch else "final"
    plt.savefig(f"generated_samples_{epoch_label}.png", bbox_inches='tight', dpi=150)
    plt.clf() 
    plt.close()
    
    del fid
    del inc
    del grid_imgs
    torch.cuda.empty_cache()
    
@torch.no_grad()
def is_score(model, schedule, device, num_samples=2048, eval_batch_size=256, steps=30, epoch=None):
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
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    BATCH_SIZE = 128
    EPOCHS = 800
    LR = 3e-4

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
    ])
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    # INCREASED NUM_WORKERS to 4 to feed the GPU faster
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)

    # FastUViT initialization
    model = FastUViT(img_size=32, patch_size=2, in_channels=3, dim=384, inner_depth=10, num_heads=12).to(device)
    
    # Optional: PyTorch 2.0 Free Speedup
    # model = torch.compile(model)
    
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
            target_v = sqrt_alpha_bar * noise - sqrt_one_minus_alpha_bar * imgs 
            
            snr_t = schedule.snr[t].view(B, 1, 1, 1)
            loss_weight = torch.clamp(snr_t, max=5.0) / (snr_t + 1.0)

            optimizer.zero_grad(set_to_none=True)
            
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

        if epoch % 10 == 0:
            curr_is = is_score(ema_model, schedule, device, num_samples=256, eval_batch_size=256, steps=30, epoch=epoch+1)
            print(f"Epoch {epoch} Mini-IS: {curr_is:.2f}")
            if curr_is > best_is:
                best_is = curr_is
                torch.save(ema_model.state_dict(), "uvit_12th_best.pt")
                
    # ema_model.load_state_dict(torch.load("uvit_12th_best.pt", map_location=device))               
    torch.save(ema_model.state_dict(), f"uvit_12th_final.pt")
    evaluate_metrics(ema_model, schedule, dataloader, device, num_samples=50_000, steps=30, epoch="final_eval_12th_50_000")
    
# poch 194/800: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 390/390 [01:43<00:00,  3.78it/s, Loss=0.0238, LR=2.82e-04]
# Epoch 195/800: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 390/390 [01:41<00:00,  3.83it/s, Loss=0.0220, LR=2.82e-04]
# Epoch 196/800: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 390/390 [01:44<00:00,  3.74it/s, Loss=0.0230, LR=2.81e-04]
# Epoch 197/800: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 390/390 [01:41<00:00,  3.83it/s, Loss=0.0252, LR=2.81e-04]
# Epoch 198/800: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 390/390 [01:42<00:00,  3.81it/s, Loss=0.0265, LR=2.81e-04]
# Epoch 199/800: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 390/390 [01:41<00:00,  3.84it/s, Loss=0.0244, LR=2.80e-04]
# Epoch 200/800: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 390/390 [01:41<00:00,  3.83it/s, Loss=0.0225, LR=2.80e-04]
# Epoch 201/800: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 390/390 [01:41<00:00,  3.83it/s, Loss=0.0236, LR=2.80e-04]
# Epoch 200 Mini-IS: 5.29 MAXIMUM
# Epoch 202/800: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 390/390 [01:45<00:00,  3.69it/s, Loss=0.0271, LR=2.79e-04]
# Epoch 203/800: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 390/390 [01:41<00:00,  3.84it/s, Loss=0.0244, LR=2.79e-04]
# Epoch 204/800: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 390/390 [01:42<00:00,  3.81it/s, Loss=0.0239, LR=2.79e-04]
# Epoch 205/800: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 390/390 [01:42<00:00,  3.79it/s, Loss=0.0213, LR=2.78e-04]
# Epoch 206/800: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 390/390 [01:42<00:00,  3.79it/s, Loss=0.0266, LR=2.78e-04]
# Epoch 207/800: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 390/390 [01:43<00:00,  3.78it/s, Loss=0.0218, LR=2.78e-04]
# Epoch 208/800: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 390/390 [01:44<00:00,  3.73it/s, Loss=0.0227, LR=2.77e-04]
# Epoch 209/800: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 390/390 [01:44<00:00,  3.72it/s, Loss=0.0263, LR=2.77e-04]
# Epoch 210/800: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 390/390 [01:40<00:00,  3.87it/s, Loss=0.0277, LR=2.77e-04]
# Epoch 211/800: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 390/390 [01:42<00:00,  3.80it/s, Loss=0.0228, LR=2.76e-04]
# Epoch 210 Mini-IS: 5.07
# Epoch 212/800: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 390/390 [01:50<00:00,  3.54it/s, Loss=0.0242, LR=2.76e-04]
# Epoch 213/800: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 390/390 [01:41<00:00,  3.86it/s, Loss=0.0237, LR=2.75e-04]
# Epoch 214/800: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 390/390 [01:41<00:00,  3.84it/s, Loss=0.0222, LR=2.75e-04]
# Epoch 215/800: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 390/390 [01:40<00:00,  3.87it/s, Loss=0.0245, LR=2.75e-04]
# Epoch 216/800: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 390/390 [01:40<00:00,  3.86it/s, Loss=0.0184, LR=2.74e-04]
# Epoch 217/800: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 390/390 [01:40<00:00,  3.88it/s, Loss=0.0212, LR=2.74e-04]
# Epoch 218/800: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 390/390 [01:41<00:00,  3.83it/s, Loss=0.0243, LR=2.74e-04]
# Epoch 219/800: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 390/390 [01:40<00:00,  3.86it/s, Loss=0.0222, LR=2.73e-04]
# Epoch 220/800: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 390/390 [01:40<00:00,  3.87it/s, Loss=0.0268, LR=2.73e-04]
# Epoch 221/800: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 390/390 [01:40<00:00,  3.87it/s, Loss=0.0226, LR=2.72e-04]
# Epoch 220 Mini-IS: 4.92
# Epoch 222/800: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 390/390 [01:44<00:00,  3.72it/s, Loss=0.0265, LR=2.72e-04]
# Epoch 223/800: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 390/390 [01:40<00:00,  3.87it/s, Loss=0.0259, LR=2.72e-04]
# Epoch 224/800: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 390/390 [01:40<00:00,  3.88it/s, Loss=0.0270, LR=2.71e-04]
# Epoch 225/800: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 390/390 [01:41<00:00,  3.84it/s, Loss=0.0224, LR=2.71e-04]
# Epoch 226/800: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 390/390 [01:40<00:00,  3.87it/s, Loss=0.0235, LR=2.71e-04]
# Epoch 227/800: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 390/390 [01:40<00:00,  3.87it/s, Loss=0.0262, LR=2.70e-04]
# Epoch 228/800: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 390/390 [01:40<00:00,  3.88it/s, Loss=0.0223, LR=2.70e-04]
# Epoch 229/800: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 390/390 [01:40<00:00,  3.88it/s, Loss=0.0223, LR=2.69e-04]
# Epoch 230/800: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 390/390 [01:40<00:00,  3.88it/s, Loss=0.0213, LR=2.69e-04]
# Epoch 231/800: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 390/390 [01:40<00:00,  3.88it/s, Loss=0.0232, LR=2.69e-04]
# Epoch 230 Mini-IS: 4.87
# Epoch 232/800: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 390/390 [02:19<00:00,  2.79it/s, Loss=0.0228, LR=2.68e-04]
# Epoch 233/800: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 390/390 [02:01<00:00,  3.21it/s, Loss=0.0206, LR=2.68e-04]
# Epoch 234/800: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 390/390 [02:01<00:00,  3.21it/s, Loss=0.0244, LR=2.67e-04]
# Epoch 235/800: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 390/390 [01:57<00:00,  3.32it/s, Loss=0.0252, LR=2.67e-04]
# Epoch 237/800: 100%|██████████████████████████████████████████████████████████████████████████████████████| 390/390 [01:43<00:00,  3.75it/s, Loss=0.0272, LR=2.66e-04]




# --- Evaluating with 50000 samples ---
# C:\Users\Dimitar Trajkov\anaconda3\Lib\site-packages\torchmetrics\utilities\prints.py:43: UserWarning: Metric `InceptionScore` will save all extracted features in buffer. For large datasets this may lead to large memory footprint.
#   warnings.warn(*args, **kwargs)
# Total Generation Time: 1836.6725 secondsbatches left))))
# FID Score: 27.6506
# Inception Score: 7.2447 ± 0.1092