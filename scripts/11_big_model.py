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
# 2. SOTA Architecture Components
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

class OverlappingConvStem(nn.Module):
    def __init__(self, in_channels=3, out_channels=384):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels // 4),
            nn.GELU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels // 4, out_channels // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.GELU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channels // 2, out_channels, kernel_size=3, stride=2, padding=1, bias=True)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
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

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x_attn = self.norm1(x) * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        x = x + gate_msa.unsqueeze(1) * self.attn(x_attn)
        x_mlp = self.norm2(x) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_mlp)
        return x

class FastDiT(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3, dim=384, depth=12, num_heads=12):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_patches = (img_size // patch_size) ** 2
        
        self.x_embedder = OverlappingConvStem(in_channels=in_channels, out_channels=dim)
        self.t_embedder = TimestepEmbedder(dim)
        
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

def evaluate_metrics(model, schedule, dataloader, device, num_samples=50000, eval_batch_size=256, steps=30, epoch=None):
    print(f"\n--- Evaluating with {num_samples} samples ---")
    fid = FrechetInceptionDistance(feature=2048).to(device)
    inc = InceptionScore().to(device)

    # ---------------------------------------------------------
    # STEP 1: Process Real Images in Chunks
    # ---------------------------------------------------------
    real_processed = 0
    couter = num_samples// eval_batch_size + 1
    for batch, _ in dataloader:
        if real_processed >= num_samples:
            break
        
        current_batch_size = batch.shape[0]
        # Trim the last batch if it overshoots
        if real_processed + current_batch_size > num_samples:
            batch = batch[:num_samples - real_processed]
        
        batch = batch.to(device)
        real_imgs_uint8 = ((batch + 1.0) * 127.5).clamp(0, 255).byte()
        
        fid.update(real_imgs_uint8, real=True)
        real_processed += batch.shape[0]
        print(f"{couter} batches left")
        couter -=1
        del batch
        del real_imgs_uint8

    # ---------------------------------------------------------
    # STEP 2: Process Fake Images in Chunks
    # ---------------------------------------------------------
    fake_processed = 0
    total_gen_time = 0.0
    fake_imgs_for_grid = [] # We need to keep a few for the plot!
    couter = num_samples// eval_batch_size + 1
    while fake_processed < num_samples:
        current_batch_size = min(eval_batch_size, num_samples - fake_processed)
        
        fake_imgs, gen_time = sample_ddim(model, schedule, current_batch_size, device, steps=steps)
        total_gen_time += gen_time
        
        fake_imgs_uint8 = ((fake_imgs + 1.0) * 127.5).clamp(0, 255).byte()
        
        fid.update(fake_imgs_uint8, real=False)
        inc.update(fake_imgs_uint8)
        
        # Save up to 32 images on the CPU for the grid before deleting the GPU tensors
        if len(fake_imgs_for_grid) < 32:
            needed = 32 - len(fake_imgs_for_grid)
            fake_imgs_for_grid.append(fake_imgs[:needed].cpu())
            
        fake_processed += current_batch_size
        print(f"{couter} batches left")
        couter -=1
        del fake_imgs
        del fake_imgs_uint8
        torch.cuda.empty_cache()

    # ---------------------------------------------------------
    # STEP 3: Compute Final Scores and Plot
    # ---------------------------------------------------------
    fid_score = fid.compute().item()
    is_score_mean, is_score_std = inc.compute()
    
    print(f"Total Generation Time: {total_gen_time:.4f} seconds")
    print(f"FID Score: {fid_score:.4f}")
    print(f"Inception Score: {is_score_mean.item():.4f} ± {is_score_std.item():.4f}")

    # Reassemble the saved images for the grid
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
    
    # Final cleanup of the massive metric models
    del fid
    del inc
    del grid_imgs
    torch.cuda.empty_cache()
    
    
    
    
def is_score(model, schedule, device, num_samples=2048, eval_batch_size=256, steps=30, epoch=None):
    inc = InceptionScore().to(device)
    
    generated = 0
    total_gen_time = 0.0
    
    # Loop until we hit the requested number of samples
    save_one_batch = False
    while generated < num_samples:
        # Make sure the last batch doesn't overshoot num_samples
        current_batch_size = min(eval_batch_size, num_samples - generated)
        
        # Generate the chunk
        fake_imgs, gen_time = sample_ddim(model, schedule, current_batch_size, device, steps=steps)
        total_gen_time += gen_time
        
        # Format for torchmetrics
        fake_imgs_uint8 = ((fake_imgs + 1.0) * 127.5).clamp(0, 255).byte()
        
        # Update the running tally
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
            
        # Aggressive cleanup for this specific chunk
        del fake_imgs
        del fake_imgs_uint8
        torch.cuda.empty_cache()

    # Calculate the final score based on all chunks
    is_score_mean, is_score_std = inc.compute()
    score = is_score_mean.item()
    
    
    # Final cleanup of the Inception network
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

    # SCALED HYPERPARAMETERS
    BATCH_SIZE = 128
    EPOCHS = 800
    LR = 3e-4

    # ADDED DATA AUGMENTATION
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
    ])
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)

    # SCALED ARCHITECTURE (~33.3M Params)
    model = FastDiT(img_size=32, patch_size=4, in_channels=3, dim=384, depth=12, num_heads=12).to(device)
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
            # Safely generates 2048 images using batches of 256
            curr_is = is_score(ema_model, schedule, device, num_samples=256, eval_batch_size=256, steps=30, epoch=epoch+1)
            print(f"Epoch {epoch} Mini-IS: {curr_is:.2f}")
            if curr_is > best_is:
                best_is = curr_is
                torch.save(ema_model.state_dict(), "big_ddpm_11th_best.pt")
                
    # ema_model.load_state_dict(torch.load("big_ddpm_11th_final.pt", map_location=device))               
    evaluate_metrics(ema_model, schedule, dataloader, device, num_samples=2048, steps=500, epoch="final_eval_11th_300samples")
    torch.save(ema_model.state_dict(), f"big_ddpm_11th_final.pt")
    
    
# Files already downloaded and verified
# Model Parameters: 33.31 M
# Epoch 1/800: Loss=0.1247, LR=1.21e-05
# Epoch 0 Mini-IS: 1.20
# Epoch 2/800: Loss=0.0664, LR=1.24e-05
# Epoch 3/800: Loss=0.0569, LR=1.30e-05
# Epoch 4/800: Loss=0.0438, LR=1.38e-05
# Epoch 5/800: Loss=0.0527, LR=1.48e-05
# Epoch 6/800: Loss=0.0442, LR=1.60e-05
# Epoch 7/800: Loss=0.0450, LR=1.74e-05
# Epoch 8/800: Loss=0.0374, LR=1.90e-05
# Epoch 9/800: Loss=0.0493, LR=2.09e-05
# Epoch 10/800: Loss=0.0326, LR=2.30e-05
# Epoch 11/800: Loss=0.0365, LR=2.52e-05
# Epoch 10 Mini-IS: 1.23
# Epoch 12/800: Loss=0.0361, LR=2.77e-05
# Epoch 13/800: Loss=0.0346, LR=3.04e-05
# Epoch 14/800: Loss=0.0368, LR=3.32e-05
# Epoch 15/800: Loss=0.0337, LR=3.63e-05
# Epoch 16/800: Loss=0.0350, LR=3.95e-05
# Epoch 17/800: Loss=0.0289, LR=4.29e-05
# Epoch 18/800: Loss=0.0291, LR=4.65e-05
# Epoch 19/800: Loss=0.0292, LR=5.03e-05
# Epoch 20/800: Loss=0.0329, LR=5.42e-05
# Epoch 21/800: Loss=0.0308, LR=5.83e-05
# Epoch 20 Mini-IS: 1.26
# Epoch 22/800: Loss=0.0307, LR=6.25e-05
# Epoch 23/800: Loss=0.0262, LR=6.69e-05
# Epoch 24/800: Loss=0.0291, LR=7.14e-05
# Epoch 25/800: Loss=0.0292, LR=7.60e-05
# Epoch 26/800: Loss=0.0289, LR=8.08e-05
# Epoch 27/800: Loss=0.0278, LR=8.56e-05
# Epoch 28/800: Loss=0.0327, LR=9.06e-05
# Epoch 29/800: Loss=0.0299, LR=9.57e-05
# Epoch 30/800: Loss=0.0289, LR=1.01e-04
# Epoch 31/800: Loss=0.0242, LR=1.06e-04
# Epoch 30 Mini-IS: 1.25
# Epoch 32/800: Loss=0.0253, LR=1.12e-04
# Epoch 33/800: Loss=0.0289, LR=1.17e-04
# Epoch 34/800: Loss=0.0303, LR=1.22e-04
# Epoch 35/800: Loss=0.0297, LR=1.28e-04
# Epoch 36/800: Loss=0.0281, LR=1.33e-04
# Epoch 37/800: Loss=0.0268, LR=1.39e-04
# Epoch 38/800: Loss=0.0264, LR=1.45e-04
# Epoch 39/800: Loss=0.0288, LR=1.50e-04
# Epoch 40/800: Loss=0.0260, LR=1.56e-04
# Epoch 41/800: Loss=0.0321, LR=1.62e-04
# Epoch 40 Mini-IS: 1.23
# Epoch 42/800: Loss=0.0250, LR=1.67e-04
# Epoch 43/800: Loss=0.0275, LR=1.73e-04
# Epoch 44/800: Loss=0.0290, LR=1.79e-04
# Epoch 45/800: Loss=0.0246, LR=1.84e-04
# Epoch 46/800: Loss=0.0263, LR=1.90e-04
# Epoch 47/800: Loss=0.0287, LR=1.95e-04
# Epoch 48/800: Loss=0.0280, LR=2.01e-04
# Epoch 49/800: Loss=0.0227, LR=2.06e-04
# Epoch 50/800: Loss=0.0251, LR=2.11e-04
# Epoch 51/800: Loss=0.0271, LR=2.16e-04
# Epoch 50 Mini-IS: 1.23
# Epoch 52/800: Loss=0.0261, LR=2.21e-04
# Epoch 53/800: Loss=0.0291, LR=2.26e-04
# Epoch 54/800: Loss=0.0278, LR=2.31e-04
# Epoch 55/800: Loss=0.0256, LR=2.36e-04
# Epoch 56/800: Loss=0.0266, LR=2.41e-04
# Epoch 57/800: Loss=0.0223, LR=2.45e-04
# Epoch 58/800: Loss=0.0255, LR=2.50e-04
# Epoch 59/800: Loss=0.0281, LR=2.54e-04
# Epoch 60/800: Loss=0.0247, LR=2.58e-04
# Epoch 61/800: Loss=0.0278, LR=2.62e-04
# Epoch 60 Mini-IS: 1.22
# Epoch 62/800: Loss=0.0264, LR=2.66e-04
# Epoch 63/800: Loss=0.0290, LR=2.69e-04
# Epoch 64/800: Loss=0.0246, LR=2.73e-04
# Epoch 65/800: Loss=0.0273, LR=2.76e-04
# Epoch 66/800: Loss=0.0272, LR=2.79e-04
# Epoch 67/800: Loss=0.0276, LR=2.82e-04
# Epoch 68/800: Loss=0.0254, LR=2.84e-04
# Epoch 69/800: Loss=0.0290, LR=2.87e-04
# Epoch 70/800: Loss=0.0249, LR=2.89e-04
# Epoch 71/800: Loss=0.0254, LR=2.91e-04
# Epoch 70 Mini-IS: 1.24
# Epoch 72/800: Loss=0.0278, LR=2.93e-04
# Epoch 73/800: Loss=0.0262, LR=2.95e-04
# Epoch 74/800: Loss=0.0243, LR=2.96e-04
# Epoch 75/800: Loss=0.0253, LR=2.97e-04
# Epoch 76/800: Loss=0.0269, LR=2.98e-04
# Epoch 77/800: Loss=0.0270, LR=2.99e-04
# Epoch 78/800: Loss=0.0254, LR=3.00e-04
# Epoch 79/800: Loss=0.0285, LR=3.00e-04
# Epoch 80/800: Loss=0.0267, LR=3.00e-04
# Epoch 81/800: Loss=0.0250, LR=3.00e-04
# Epoch 80 Mini-IS: 1.50
# Epoch 82/800: Loss=0.0240, LR=3.00e-04
# Epoch 83/800: Loss=0.0229, LR=3.00e-04
# Epoch 84/800: Loss=0.0255, LR=3.00e-04
# Epoch 85/800: Loss=0.0249, LR=3.00e-04
# Epoch 86/800: Loss=0.0249, LR=3.00e-04
# Epoch 87/800: Loss=0.0241, LR=3.00e-04
# Epoch 88/800: Loss=0.0234, LR=3.00e-04
# Epoch 89/800: Loss=0.0267, LR=3.00e-04
# Epoch 90/800: Loss=0.0260, LR=3.00e-04
# Epoch 91/800: Loss=0.0314, LR=3.00e-04
# Epoch 90 Mini-IS: 1.84
# Epoch 92/800: Loss=0.0292, LR=3.00e-04
# Epoch 93/800: Loss=0.0235, LR=3.00e-04
# Epoch 94/800: Loss=0.0239, LR=3.00e-04
# Epoch 95/800: Loss=0.0265, LR=3.00e-04
# Epoch 96/800: Loss=0.0212, LR=3.00e-04
# Epoch 97/800: Loss=0.0272, LR=3.00e-04
# Epoch 98/800: Loss=0.0225, LR=3.00e-04
# Epoch 99/800: Loss=0.0260, LR=2.99e-04
# Epoch 100/800: Loss=0.0269, LR=2.99e-04
# Epoch 101/800: Loss=0.0257, LR=2.99e-04
# Epoch 100 Mini-IS: 2.12
# Epoch 102/800: Loss=0.0239, LR=2.99e-04
# Epoch 103/800: Loss=0.0259, LR=2.99e-04
# Epoch 104/800: Loss=0.0264, LR=2.99e-04
# Epoch 105/800: Loss=0.0248, LR=2.99e-04
# Epoch 106/800: Loss=0.0250, LR=2.99e-04
# Epoch 107/800: Loss=0.0273, LR=2.99e-04
# Epoch 108/800: Loss=0.0220, LR=2.99e-04
# Epoch 109/800: Loss=0.0251, LR=2.99e-04
# Epoch 110/800: Loss=0.0277, LR=2.99e-04
# Epoch 111/800: Loss=0.0252, LR=2.99e-04
# Epoch 110 Mini-IS: 2.62
# Epoch 112/800: Loss=0.0278, LR=2.99e-04
# Epoch 113/800: Loss=0.0226, LR=2.98e-04
# Epoch 114/800: Loss=0.0241, LR=2.98e-04
# Epoch 115/800: Loss=0.0232, LR=2.98e-04
# Epoch 116/800: Loss=0.0241, LR=2.98e-04
# Epoch 117/800: Loss=0.0241, LR=2.98e-04
# Epoch 118/800: Loss=0.0237, LR=2.98e-04
# Epoch 119/800: Loss=0.0241, LR=2.98e-04
# Epoch 120/800: Loss=0.0241, LR=2.98e-04
# Epoch 121/800: Loss=0.0282, LR=2.98e-04
# Epoch 120 Mini-IS: 3.06
# Epoch 122/800: Loss=0.0270, LR=2.97e-04
# Epoch 123/800: Loss=0.0254, LR=2.97e-04
# Epoch 124/800: Loss=0.0235, LR=2.97e-04
# Epoch 125/800: Loss=0.0258, LR=2.97e-04
# Epoch 126/800: Loss=0.0270, LR=2.97e-04
# Epoch 127/800: Loss=0.0240, LR=2.97e-04
# Epoch 128/800: Loss=0.0250, LR=2.97e-04
# Epoch 129/800: Loss=0.0251, LR=2.97e-04
# Epoch 130/800: Loss=0.0232, LR=2.96e-04
# Epoch 131/800: Loss=0.0232, LR=2.96e-04
# Epoch 130 Mini-IS: 3.49
# Epoch 132/800: Loss=0.0199, LR=2.96e-04
# Epoch 133/800: Loss=0.0270, LR=2.96e-04
# Epoch 134/800: Loss=0.0251, LR=2.96e-04
# Epoch 135/800: Loss=0.0258, LR=2.96e-04
# Epoch 136/800: Loss=0.0251, LR=2.96e-04
# Epoch 137/800: Loss=0.0226, LR=2.95e-04
# Epoch 138/800: Loss=0.0247, LR=2.95e-04
# Epoch 139/800: Loss=0.0265, LR=2.95e-04
# Epoch 140/800: Loss=0.0253, LR=2.95e-04
# Epoch 141/800: Loss=0.0252, LR=2.95e-04
# Epoch 140 Mini-IS: 3.90
# Epoch 142/800: Loss=0.0295, LR=2.95e-04
# Epoch 143/800: Loss=0.0237, LR=2.94e-04
# Epoch 144/800: Loss=0.0265, LR=2.94e-04
# Epoch 145/800: Loss=0.0230, LR=2.94e-04
# Epoch 146/800: Loss=0.0278, LR=2.94e-04
# Epoch 147/800: Loss=0.0243, LR=2.94e-04
# Epoch 148/800: Loss=0.0233, LR=2.93e-04
# Epoch 149/800: Loss=0.0254, LR=2.93e-04
# Epoch 150/800: Loss=0.0243, LR=2.93e-04
# Epoch 151/800: Loss=0.0257, LR=2.93e-04
# Epoch 150 Mini-IS: 4.30
# Epoch 152/800: Loss=0.0263, LR=2.93e-04
# Epoch 153/800: Loss=0.0240, LR=2.92e-04
# Epoch 154/800: Loss=0.0261, LR=2.92e-04
# Epoch 155/800: Loss=0.0253, LR=2.92e-04
# Epoch 156/800: Loss=0.0278, LR=2.92e-04
# Epoch 157/800: Loss=0.0231, LR=2.92e-04
# Epoch 158/800: Loss=0.0277, LR=2.91e-04
# Epoch 159/800: Loss=0.0244, LR=2.91e-04
# Epoch 160/800: Loss=0.0281, LR=2.91e-04
# Epoch 161/800: Loss=0.0244, LR=2.91e-04
# Epoch 160 Mini-IS: 4.66
# Epoch 162/800: Loss=0.0263, LR=2.91e-04
# Epoch 163/800: Loss=0.0262, LR=2.90e-04
# Epoch 164/800: Loss=0.0226, LR=2.90e-04
# Epoch 165/800: Loss=0.0236, LR=2.90e-04
# Epoch 166/800: Loss=0.0218, LR=2.90e-04
# Epoch 167/800: Loss=0.0247, LR=2.89e-04
# Epoch 168/800: Loss=0.0224, LR=2.89e-04
# Epoch 169/800: Loss=0.0255, LR=2.89e-04
# Epoch 170/800: Loss=0.0232, LR=2.89e-04
# Epoch 171/800: Loss=0.0266, LR=2.88e-04
# Epoch 170 Mini-IS: 4.74
# Epoch 172/800: Loss=0.0262, LR=2.88e-04
# Epoch 173/800: Loss=0.0291, LR=2.88e-04
# Epoch 174/800: Loss=0.0257, LR=2.88e-04
# Epoch 175/800: Loss=0.0258, LR=2.87e-04
# Epoch 176/800: Loss=0.0301, LR=2.87e-04
# Epoch 177/800: Loss=0.0257, LR=2.87e-04
# Epoch 178/800: Loss=0.0258, LR=2.86e-04
# Epoch 179/800: Loss=0.0250, LR=2.86e-04
# Epoch 180/800: Loss=0.0264, LR=2.86e-04
# Epoch 181/800: Loss=0.0261, LR=2.86e-04
# Epoch 180 Mini-IS: 4.73
# Epoch 182/800: Loss=0.0239, LR=2.85e-04
# Epoch 183/800: Loss=0.0261, LR=2.85e-04
# Epoch 184/800: Loss=0.0247, LR=2.85e-04
# Epoch 185/800: Loss=0.0233, LR=2.85e-04
# Epoch 186/800: Loss=0.0262, LR=2.84e-04
# Epoch 187/800: Loss=0.0266, LR=2.84e-04
# Epoch 188/800: Loss=0.0249, LR=2.84e-04
# Epoch 189/800: Loss=0.0248, LR=2.83e-04
# Epoch 190/800: Loss=0.0251, LR=2.83e-04
# Epoch 191/800: Loss=0.0242, LR=2.83e-04
# Epoch 190 Mini-IS: 4.93
# Epoch 192/800: Loss=0.0257, LR=2.82e-04
# Epoch 193/800: Loss=0.0230, LR=2.82e-04
# Epoch 194/800: Loss=0.0213, LR=2.82e-04
# Epoch 195/800: Loss=0.0255, LR=2.82e-04
# Epoch 196/800: Loss=0.0238, LR=2.81e-04
# Epoch 197/800: Loss=0.0235, LR=2.81e-04
# Epoch 198/800: Loss=0.0249, LR=2.81e-04
# Epoch 199/800: Loss=0.0245, LR=2.80e-04
# Epoch 200/800: Loss=0.0246, LR=2.80e-04
# Epoch 201/800: Loss=0.0191, LR=2.80e-04
# Epoch 200 Mini-IS: 5.05
# Epoch 202/800: Loss=0.0277, LR=2.79e-04
# Epoch 203/800: Loss=0.0238, LR=2.79e-04
# Epoch 204/800: Loss=0.0216, LR=2.79e-04
# Epoch 205/800: Loss=0.0226, LR=2.78e-04
# Epoch 206/800: Loss=0.0256, LR=2.78e-04
# Epoch 207/800: Loss=0.0253, LR=2.78e-04
# Epoch 208/800: Loss=0.0203, LR=2.77e-04
# Epoch 209/800: Loss=0.0218, LR=2.77e-04
# Epoch 210/800: Loss=0.0228, LR=2.77e-04
# Epoch 211/800: Loss=0.0242, LR=2.76e-04
# Epoch 210 Mini-IS: 5.06
# Epoch 212/800: Loss=0.0273, LR=2.76e-04
# Epoch 213/800: Loss=0.0249, LR=2.75e-04
# Epoch 214/800: Loss=0.0217, LR=2.75e-04
# Epoch 215/800: Loss=0.0209, LR=2.75e-04
# Epoch 216/800: Loss=0.0250, LR=2.74e-04
# Epoch 217/800: Loss=0.0226, LR=2.74e-04
# Epoch 218/800: Loss=0.0217, LR=2.74e-04
# Epoch 219/800: Loss=0.0246, LR=2.73e-04
# Epoch 220/800: Loss=0.0238, LR=2.73e-04
# Epoch 221/800: Loss=0.0259, LR=2.72e-04
# Epoch 220 Mini-IS: 5.48
# Epoch 222/800: Loss=0.0228, LR=2.72e-04
# Epoch 223/800: Loss=0.0285, LR=2.72e-04
# Epoch 224/800: Loss=0.0274, LR=2.71e-04
# Epoch 225/800: Loss=0.0261, LR=2.71e-04
# Epoch 226/800: Loss=0.0253, LR=2.71e-04
# Epoch 227/800: Loss=0.0261, LR=2.70e-04
# Epoch 228/800: Loss=0.0265, LR=2.70e-04
# Epoch 229/800: Loss=0.0212, LR=2.69e-04
# Epoch 230/800: Loss=0.0247, LR=2.69e-04
# Epoch 231/800: Loss=0.0255, LR=2.69e-04
# Epoch 230 Mini-IS: 5.02
# Epoch 232/800: Loss=0.0223, LR=2.68e-04
# Epoch 233/800: Loss=0.0230, LR=2.68e-04
# Epoch 234/800: Loss=0.0231, LR=2.67e-04
# Epoch 235/800: Loss=0.0227, LR=2.67e-04
# Epoch 236/800: Loss=0.0242, LR=2.67e-04
# Epoch 237/800: Loss=0.0242, LR=2.66e-04
# Epoch 238/800: Loss=0.0239, LR=2.66e-04
# Epoch 239/800: Loss=0.0240, LR=2.65e-04
# Epoch 240/800: Loss=0.0236, LR=2.65e-04
# Epoch 241/800: Loss=0.0240, LR=2.64e-04
# Epoch 240 Mini-IS: 5.28
# Epoch 242/800: Loss=0.0251, LR=2.64e-04
# Epoch 243/800: Loss=0.0269, LR=2.64e-04
# Epoch 244/800: Loss=0.0268, LR=2.63e-04
# Epoch 245/800: Loss=0.0248, LR=2.63e-04
# Epoch 246/800: Loss=0.0233, LR=2.62e-04
# Epoch 247/800: Loss=0.0243, LR=2.62e-04
# Epoch 248/800: Loss=0.0265, LR=2.61e-04
# Epoch 249/800: Loss=0.0214, LR=2.61e-04
# Epoch 250/800: Loss=0.0250, LR=2.61e-04
# Epoch 251/800: Loss=0.0226, LR=2.60e-04
# Epoch 250 Mini-IS: 5.03
# Epoch 252/800: Loss=0.0251, LR=2.60e-04
# Epoch 253/800: Loss=0.0251, LR=2.59e-04
# Epoch 254/800: Loss=0.0236, LR=2.59e-04
# Epoch 255/800: Loss=0.0243, LR=2.58e-04
# Epoch 256/800: Loss=0.0252, LR=2.58e-04
# Epoch 257/800: Loss=0.0245, LR=2.57e-04
# Epoch 258/800: Loss=0.0259, LR=2.57e-04
# Epoch 259/800: Loss=0.0261, LR=2.57e-04
# Epoch 260/800: Loss=0.0289, LR=2.56e-04
# Epoch 261/800: Loss=0.0242, LR=2.56e-04
# Epoch 260 Mini-IS: 5.02
# Epoch 262/800: Loss=0.0253, LR=2.55e-04
# Epoch 263/800: Loss=0.0278, LR=2.55e-04
# Epoch 264/800: Loss=0.0261, LR=2.54e-04
# Epoch 265/800: Loss=0.0238, LR=2.54e-04
# Epoch 266/800: Loss=0.0237, LR=2.53e-04
# Epoch 267/800: Loss=0.0223, LR=2.53e-04
# Epoch 268/800: Loss=0.0265, LR=2.52e-04
# Epoch 269/800: Loss=0.0228, LR=2.52e-04
# Epoch 270/800: Loss=0.0264, LR=2.51e-04
# Epoch 271/800: Loss=0.0243, LR=2.51e-04
# Epoch 270 Mini-IS: 4.79
# Epoch 272/800: Loss=0.0220, LR=2.50e-04
# Epoch 273/800: Loss=0.0240, LR=2.50e-04
# Epoch 274/800: Loss=0.0219, LR=2.49e-04
# Epoch 275/800: Loss=0.0230, LR=2.49e-04
# Epoch 276/800: Loss=0.0215, LR=2.48e-04
# Epoch 277/800: Loss=0.0240, LR=2.48e-04
# Epoch 278/800: Loss=0.0223, LR=2.47e-04
# Epoch 279/800: Loss=0.0253, LR=2.47e-04
# Epoch 280/800: Loss=0.0254, LR=2.46e-04
# Epoch 281/800: Loss=0.0229, LR=2.46e-04
# Epoch 280 Mini-IS: 5.13
# Epoch 282/800: Loss=0.0207, LR=2.45e-04
# Epoch 283/800: Loss=0.0225, LR=2.45e-04
# Epoch 284/800: Loss=0.0233, LR=2.44e-04
# Epoch 285/800: Loss=0.0272, LR=2.44e-04
# Epoch 286/800: Loss=0.0239, LR=2.43e-04
# Epoch 287/800: Loss=0.0268, LR=2.43e-04
# Epoch 288/800: Loss=0.0269, LR=2.42e-04
# Epoch 289/800: Loss=0.0223, LR=2.42e-04
# Epoch 290/800: Loss=0.0260, LR=2.41e-04
# Epoch 291/800: Loss=0.0239, LR=2.41e-04
# Epoch 290 Mini-IS: 5.17
# Epoch 292/800: Loss=0.0220, LR=2.40e-04
# Epoch 293/800: Loss=0.0208, LR=2.40e-04
# Epoch 294/800: Loss=0.0261, LR=2.39e-04
# Epoch 295/800: Loss=0.0248, LR=2.39e-04
# Epoch 296/800: Loss=0.0218, LR=2.38e-04
# Epoch 297/800: Loss=0.0237, LR=2.38e-04
# Epoch 298/800: Loss=0.0219, LR=2.37e-04
# Epoch 299/800: Loss=0.0207, LR=2.37e-04
# Epoch 300/800: Loss=0.0225, LR=2.36e-04
# Epoch 301/800: Loss=0.0251, LR=2.35e-04
# Epoch 300 Mini-IS: 5.38
# Epoch 302/800: Loss=0.0221, LR=2.35e-04
# Epoch 303/800: Loss=0.0238, LR=2.34e-04
# Epoch 304/800: Loss=0.0233, LR=2.34e-04
# Epoch 305/800: Loss=0.0259, LR=2.33e-04
# Epoch 306/800: Loss=0.0239, LR=2.33e-04
# Epoch 307/800: Loss=0.0243, LR=2.32e-04
# Epoch 308/800: Loss=0.0248, LR=2.32e-04
# Epoch 309/800: Loss=0.0258, LR=2.31e-04
# Epoch 310/800: Loss=0.0248, LR=2.31e-04
# Epoch 311/800: Loss=0.0245, LR=2.30e-04
# Epoch 310 Mini-IS: 5.43
# Epoch 312/800: Loss=0.0231, LR=2.29e-04
# Epoch 313/800: Loss=0.0258, LR=2.29e-04
# Epoch 314/800: Loss=0.0261, LR=2.28e-04
# Epoch 315/800: Loss=0.0224, LR=2.28e-04
# Epoch 316/800: Loss=0.0251, LR=2.27e-04
# Epoch 317/800: Loss=0.0248, LR=2.27e-04
# Epoch 318/800: Loss=0.0203, LR=2.26e-04
# Epoch 319/800: Loss=0.0262, LR=2.26e-04
# Epoch 320/800: Loss=0.0214, LR=2.25e-04
# Epoch 321/800: Loss=0.0238, LR=2.24e-04
# Epoch 320 Mini-IS: 5.18
# Epoch 322/800: Loss=0.0251, LR=2.24e-04
# Epoch 323/800: Loss=0.0247, LR=2.23e-04
# Epoch 324/800: Loss=0.0239, LR=2.23e-04
# Epoch 325/800: Loss=0.0208, LR=2.22e-04
# Epoch 326/800: Loss=0.0229, LR=2.22e-04
# Epoch 327/800: Loss=0.0217, LR=2.21e-04
# Epoch 328/800: Loss=0.0240, LR=2.20e-04
# Epoch 329/800: Loss=0.0242, LR=2.20e-04
# Epoch 330/800: Loss=0.0261, LR=2.19e-04
# Epoch 331/800: Loss=0.0226, LR=2.19e-04
# Epoch 330 Mini-IS: 5.51
# Epoch 332/800: Loss=0.0222, LR=2.18e-04
# Epoch 333/800: Loss=0.0251, LR=2.18e-04
# Epoch 334/800: Loss=0.0277, LR=2.17e-04
# Epoch 335/800: Loss=0.0254, LR=2.16e-04
# Epoch 336/800: Loss=0.0257, LR=2.16e-04
# Epoch 337/800: Loss=0.0242, LR=2.15e-04
# Epoch 338/800: Loss=0.0265, LR=2.15e-04
# Epoch 339/800: Loss=0.0229, LR=2.14e-04
# Epoch 340/800: Loss=0.0217, LR=2.13e-04
# Epoch 341/800: Loss=0.0216, LR=2.13e-04
# Epoch 340 Mini-IS: 5.14
# Epoch 342/800: Loss=0.0262, LR=2.12e-04
# Epoch 343/800: Loss=0.0239, LR=2.12e-04
# Epoch 344/800: Loss=0.0227, LR=2.11e-04
# Epoch 345/800: Loss=0.0270, LR=2.10e-04
# Epoch 346/800: Loss=0.0278, LR=2.10e-04
# Epoch 347/800: Loss=0.0209, LR=2.09e-04
# Epoch 348/800: Loss=0.0226, LR=2.09e-04
# Epoch 349/800: Loss=0.0252, LR=2.08e-04
# Epoch 350/800: Loss=0.0253, LR=2.07e-04
# Epoch 351/800: Loss=0.0240, LR=2.07e-04
# Epoch 350 Mini-IS: 5.48
# Epoch 352/800: Loss=0.0241, LR=2.06e-04
# Epoch 353/800: Loss=0.0222, LR=2.06e-04
# Epoch 354/800: Loss=0.0240, LR=2.05e-04
# Epoch 355/800: Loss=0.0272, LR=2.04e-04
# Epoch 356/800: Loss=0.0225, LR=2.04e-04
# Epoch 357/800: Loss=0.0225, LR=2.03e-04
# Epoch 358/800: Loss=0.0254, LR=2.03e-04
# Epoch 359/800: Loss=0.0226, LR=2.02e-04
# Epoch 360/800: Loss=0.0258, LR=2.01e-04
# Epoch 361/800: Loss=0.0247, LR=2.01e-04
# Epoch 360 Mini-IS: 5.44
# Epoch 362/800: Loss=0.0225, LR=2.00e-04
# Epoch 363/800: Loss=0.0250, LR=1.99e-04
# Epoch 364/800: Loss=0.0224, LR=1.99e-04
# Epoch 365/800: Loss=0.0227, LR=1.98e-04
# Epoch 366/800: Loss=0.0212, LR=1.98e-04
# Epoch 367/800: Loss=0.0244, LR=1.97e-04
# Epoch 368/800: Loss=0.0226, LR=1.96e-04
# Epoch 369/800: Loss=0.0233, LR=1.96e-04
# Epoch 370/800: Loss=0.0236, LR=1.95e-04
# Epoch 371/800: Loss=0.0233, LR=1.94e-04
# Epoch 370 Mini-IS: 5.06
# Epoch 372/800: Loss=0.0222, LR=1.94e-04
# Epoch 373/800: Loss=0.0231, LR=1.93e-04
# Epoch 374/800: Loss=0.0246, LR=1.93e-04
# Epoch 375/800: Loss=0.0256, LR=1.92e-04
# Epoch 376/800: Loss=0.0228, LR=1.91e-04
# Epoch 377/800: Loss=0.0251, LR=1.91e-04
# Epoch 378/800: Loss=0.0252, LR=1.90e-04
# Epoch 379/800: Loss=0.0210, LR=1.89e-04
# Epoch 380/800: Loss=0.0280, LR=1.89e-04
# Epoch 381/800: Loss=0.0248, LR=1.88e-04
# Epoch 380 Mini-IS: 5.23
# Epoch 382/800: Loss=0.0205, LR=1.88e-04
# Epoch 383/800: Loss=0.0245, LR=1.87e-04
# Epoch 384/800: Loss=0.0224, LR=1.86e-04
# Epoch 385/800: Loss=0.0240, LR=1.86e-04
# Epoch 386/800: Loss=0.0253, LR=1.85e-04
# Epoch 387/800: Loss=0.0247, LR=1.84e-04
# Epoch 388/800: Loss=0.0251, LR=1.84e-04
# Epoch 389/800: Loss=0.0205, LR=1.83e-04
# Epoch 390/800: Loss=0.0230, LR=1.82e-04
# Epoch 391/800: Loss=0.0234, LR=1.82e-04
# Epoch 390 Mini-IS: 5.32
# Epoch 392/800: Loss=0.0260, LR=1.81e-04
# Epoch 393/800: Loss=0.0200, LR=1.81e-04
# Epoch 394/800: Loss=0.0271, LR=1.80e-04
# Epoch 395/800: Loss=0.0254, LR=1.79e-04
# Epoch 396/800: Loss=0.0235, LR=1.79e-04
# Epoch 397/800: Loss=0.0223, LR=1.78e-04
# Epoch 398/800: Loss=0.0249, LR=1.77e-04
# Epoch 399/800: Loss=0.0247, LR=1.77e-04
# Epoch 400/800: Loss=0.0241, LR=1.76e-04
# Epoch 401/800: Loss=0.0231, LR=1.75e-04
# Epoch 400 Mini-IS: 5.17
# Epoch 402/800: Loss=0.0255, LR=1.75e-04
# Epoch 403/800: Loss=0.0232, LR=1.74e-04
# Epoch 404/800: Loss=0.0239, LR=1.73e-04
# Epoch 405/800: Loss=0.0220, LR=1.73e-04
# Epoch 406/800: Loss=0.0206, LR=1.72e-04
# Epoch 407/800: Loss=0.0256, LR=1.72e-04
# Epoch 408/800: Loss=0.0219, LR=1.71e-04
# Epoch 409/800: Loss=0.0270, LR=1.70e-04
# Epoch 410/800: Loss=0.0253, LR=1.70e-04
# Epoch 411/800: Loss=0.0228, LR=1.69e-04
# Epoch 410 Mini-IS: 5.13
# Epoch 412/800: Loss=0.0271, LR=1.68e-04
# Epoch 413/800: Loss=0.0252, LR=1.68e-04
# Epoch 414/800: Loss=0.0226, LR=1.67e-04
# Epoch 415/800: Loss=0.0218, LR=1.66e-04
# Epoch 416/800: Loss=0.0218, LR=1.66e-04
# Epoch 417/800: Loss=0.0243, LR=1.65e-04
# Epoch 418/800: Loss=0.0232, LR=1.64e-04
# Epoch 419/800: Loss=0.0232, LR=1.64e-04
# Epoch 420/800: Loss=0.0233, LR=1.63e-04
# Epoch 421/800: Loss=0.0213, LR=1.62e-04
# Epoch 420 Mini-IS: 4.75
# Epoch 422/800: Loss=0.0253, LR=1.62e-04
# Epoch 423/800: Loss=0.0242, LR=1.61e-04
# Epoch 424/800: Loss=0.0233, LR=1.60e-04
# Epoch 425/800: Loss=0.0252, LR=1.60e-04
# Epoch 426/800: Loss=0.0258, LR=1.59e-04
# Epoch 427/800: Loss=0.0241, LR=1.59e-04
# Epoch 428/800: Loss=0.0248, LR=1.58e-04
# Epoch 429/800: Loss=0.0267, LR=1.57e-04
# Epoch 430/800: Loss=0.0199, LR=1.57e-04
# Epoch 431/800: Loss=0.0227, LR=1.56e-04
# Epoch 430 Mini-IS: 5.22
# Epoch 432/800: Loss=0.0231, LR=1.55e-04
# Epoch 433/800: Loss=0.0213, LR=1.55e-04
# Epoch 434/800: Loss=0.0251, LR=1.54e-04
# Epoch 435/800: Loss=0.0235, LR=1.53e-04
# Epoch 436/800: Loss=0.0239, LR=1.53e-04
# Epoch 437/800: Loss=0.0243, LR=1.52e-04
# Epoch 438/800: Loss=0.0260, LR=1.51e-04
# Epoch 439/800: Loss=0.0223, LR=1.51e-04
# Epoch 440/800: Loss=0.0220, LR=1.50e-04
# Epoch 441/800: Loss=0.0250, LR=1.49e-04
# Epoch 440 Mini-IS: 5.36
# Epoch 442/800: Loss=0.0232, LR=1.49e-04
# Epoch 443/800: Loss=0.0269, LR=1.48e-04
# Epoch 444/800: Loss=0.0245, LR=1.47e-04
# Epoch 445/800: Loss=0.0217, LR=1.47e-04
# Epoch 446/800: Loss=0.0202, LR=1.46e-04
# Epoch 447/800: Loss=0.0254, LR=1.45e-04
# Epoch 448/800: Loss=0.0226, LR=1.45e-04
# Epoch 449/800: Loss=0.0251, LR=1.44e-04
# Epoch 450/800: Loss=0.0222, LR=1.43e-04
# Epoch 451/800: Loss=0.0231, LR=1.43e-04
# Epoch 450 Mini-IS: 5.44
# Epoch 452/800: Loss=0.0247, LR=1.42e-04
# Epoch 453/800: Loss=0.0213, LR=1.41e-04
# Epoch 454/800: Loss=0.0233, LR=1.41e-04
# Epoch 455/800: Loss=0.0253, LR=1.40e-04
# Epoch 456/800: Loss=0.0218, LR=1.40e-04
# Epoch 457/800: Loss=0.0201, LR=1.39e-04
# Epoch 458/800: Loss=0.0218, LR=1.38e-04
# Epoch 459/800: Loss=0.0235, LR=1.38e-04
# Epoch 460/800: Loss=0.0218, LR=1.37e-04
# Epoch 461/800: Loss=0.0227, LR=1.36e-04
# Epoch 460 Mini-IS: 5.38
# Epoch 462/800: Loss=0.0249, LR=1.36e-04
# Epoch 463/800: Loss=0.0217, LR=1.35e-04
# Epoch 464/800: Loss=0.0230, LR=1.34e-04
# Epoch 465/800: Loss=0.0232, LR=1.34e-04
# Epoch 466/800: Loss=0.0228, LR=1.33e-04
# Epoch 467/800: Loss=0.0242, LR=1.32e-04
# Epoch 468/800: Loss=0.0207, LR=1.32e-04
# Epoch 469/800: Loss=0.0234, LR=1.31e-04
# Epoch 470/800: Loss=0.0242, LR=1.30e-04
# Epoch 471/800: Loss=0.0239, LR=1.30e-04
# Epoch 470 Mini-IS: 4.90
# Epoch 472/800: Loss=0.0220, LR=1.29e-04
# Epoch 473/800: Loss=0.0274, LR=1.28e-04
# Epoch 474/800: Loss=0.0246, LR=1.28e-04
# Epoch 475/800: Loss=0.0248, LR=1.27e-04
# Epoch 476/800: Loss=0.0234, LR=1.27e-04
# Epoch 477/800: Loss=0.0225, LR=1.26e-04
# Epoch 478/800: Loss=0.0261, LR=1.25e-04
# Epoch 479/800: Loss=0.0235, LR=1.25e-04
# Epoch 480/800: Loss=0.0214, LR=1.24e-04
# Epoch 481/800: Loss=0.0227, LR=1.23e-04
# Epoch 480 Mini-IS: 5.43
# Epoch 482/800: Loss=0.0224, LR=1.23e-04
# Epoch 483/800: Loss=0.0244, LR=1.22e-04
# Epoch 484/800: Loss=0.0214, LR=1.21e-04
# Epoch 485/800: Loss=0.0221, LR=1.21e-04
# Epoch 486/800: Loss=0.0231, LR=1.20e-04
# Epoch 487/800: Loss=0.0223, LR=1.19e-04
# Epoch 488/800: Loss=0.0251, LR=1.19e-04
# Epoch 489/800: Loss=0.0264, LR=1.18e-04
# Epoch 490/800: Loss=0.0250, LR=1.18e-04
# Epoch 491/800: Loss=0.0226, LR=1.17e-04
# Epoch 490 Mini-IS: 5.22
# Epoch 492/800: Loss=0.0215, LR=1.16e-04
# Epoch 493/800: Loss=0.0229, LR=1.16e-04
# Epoch 494/800: Loss=0.0225, LR=1.15e-04
# Epoch 495/800: Loss=0.0216, LR=1.14e-04
# Epoch 496/800: Loss=0.0230, LR=1.14e-04
# Epoch 497/800: Loss=0.0236, LR=1.13e-04
# Epoch 498/800: Loss=0.0253, LR=1.12e-04
# Epoch 499/800: Loss=0.0226, LR=1.12e-04
# Epoch 500/800: Loss=0.0233, LR=1.11e-04
# Epoch 501/800: Loss=0.0218, LR=1.11e-04
# Epoch 500 Mini-IS: 5.35
# Epoch 502/800: Loss=0.0243, LR=1.10e-04
# Epoch 503/800: Loss=0.0245, LR=1.09e-04
# Epoch 504/800: Loss=0.0221, LR=1.09e-04
# Epoch 505/800: Loss=0.0223, LR=1.08e-04
# Epoch 506/800: Loss=0.0244, LR=1.07e-04
# Epoch 507/800: Loss=0.0254, LR=1.07e-04
# Epoch 508/800: Loss=0.0254, LR=1.06e-04
# Epoch 509/800: Loss=0.0223, LR=1.06e-04
# Epoch 510/800: Loss=0.0226, LR=1.05e-04
# Epoch 511/800: Loss=0.0209, LR=1.04e-04
# Epoch 510 Mini-IS: 5.25
# Epoch 512/800: Loss=0.0251, LR=1.04e-04
# Epoch 513/800: Loss=0.0225, LR=1.03e-04
# Epoch 514/800: Loss=0.0242, LR=1.02e-04
# Epoch 515/800: Loss=0.0245, LR=1.02e-04
# Epoch 516/800: Loss=0.0239, LR=1.01e-04
# Epoch 517/800: Loss=0.0225, LR=1.01e-04
# Epoch 518/800: Loss=0.0260, LR=9.99e-05
# Epoch 519/800: Loss=0.0209, LR=9.93e-05
# Epoch 520/800: Loss=0.0207, LR=9.87e-05
# Epoch 521/800: Loss=0.0244, LR=9.81e-05
# Epoch 520 Mini-IS: 5.28
# Epoch 522/800: Loss=0.0231, LR=9.75e-05
# Epoch 523/800: Loss=0.0204, LR=9.69e-05
# Epoch 524/800: Loss=0.0219, LR=9.62e-05
# Epoch 525/800: Loss=0.0219, LR=9.56e-05
# Epoch 526/800: Loss=0.0237, LR=9.50e-05
# Epoch 527/800: Loss=0.0261, LR=9.44e-05
# Epoch 528/800: Loss=0.0227, LR=9.38e-05
# Epoch 529/800: Loss=0.0224, LR=9.32e-05
# Epoch 530/800: Loss=0.0221, LR=9.26e-05
# Epoch 531/800: Loss=0.0234, LR=9.20e-05
# Epoch 530 Mini-IS: 5.17
# Epoch 532/800: Loss=0.0211, LR=9.14e-05
# Epoch 533/800: Loss=0.0248, LR=9.08e-05
# Epoch 534/800: Loss=0.0203, LR=9.02e-05
# Epoch 535/800: Loss=0.0215, LR=8.96e-05
# Epoch 536/800: Loss=0.0220, LR=8.90e-05
# Epoch 537/800: Loss=0.0224, LR=8.84e-05
# Epoch 538/800: Loss=0.0250, LR=8.78e-05
# Epoch 539/800: Loss=0.0201, LR=8.72e-05
# Epoch 540/800: Loss=0.0224, LR=8.66e-05
# Epoch 541/800: Loss=0.0222, LR=8.60e-05
# Epoch 540 Mini-IS: 5.50
# Epoch 542/800: Loss=0.0237, LR=8.54e-05
# Epoch 543/800: Loss=0.0253, LR=8.48e-05
# Epoch 544/800: Loss=0.0230, LR=8.42e-05
# Epoch 545/800: Loss=0.0208, LR=8.37e-05
# Epoch 546/800: Loss=0.0236, LR=8.31e-05
# Epoch 547/800: Loss=0.0254, LR=8.25e-05
# Epoch 548/800: Loss=0.0210, LR=8.19e-05
# Epoch 549/800: Loss=0.0227, LR=8.13e-05
# Epoch 550/800: Loss=0.0218, LR=8.07e-05
# Epoch 551/800: Loss=0.0225, LR=8.02e-05
# Epoch 550 Mini-IS: 5.09
# Epoch 552/800: Loss=0.0248, LR=7.96e-05
# Epoch 553/800: Loss=0.0212, LR=7.90e-05
# Epoch 554/800: Loss=0.0238, LR=7.84e-05
# Epoch 555/800: Loss=0.0238, LR=7.79e-05
# Epoch 556/800: Loss=0.0210, LR=7.73e-05
# Epoch 557/800: Loss=0.0282, LR=7.67e-05
# Epoch 558/800: Loss=0.0229, LR=7.61e-05
# Epoch 559/800: Loss=0.0271, LR=7.56e-05
# Epoch 560/800: Loss=0.0217, LR=7.50e-05
# Epoch 561/800: Loss=0.0192, LR=7.44e-05
# Epoch 560 Mini-IS: 5.68
# Epoch 562/800: Loss=0.0237, LR=7.39e-05
# Epoch 563/800: Loss=0.0195, LR=7.33e-05
# Epoch 564/800: Loss=0.0232, LR=7.27e-05
# Epoch 565/800: Loss=0.0222, LR=7.22e-05
# Epoch 566/800: Loss=0.0230, LR=7.16e-05
# Epoch 567/800: Loss=0.0201, LR=7.11e-05
# Epoch 568/800: Loss=0.0232, LR=7.05e-05
# Epoch 569/800: Loss=0.0236, LR=7.00e-05
# Epoch 570/800: Loss=0.0235, LR=6.94e-05
# Epoch 571/800: Loss=0.0249, LR=6.89e-05
# Epoch 570 Mini-IS: 5.26
# Epoch 572/800: Loss=0.0212, LR=6.83e-05
# Epoch 573/800: Loss=0.0213, LR=6.78e-05
# Epoch 574/800: Loss=0.0231, LR=6.72e-05
# Epoch 575/800: Loss=0.0246, LR=6.67e-05
# Epoch 576/800: Loss=0.0203, LR=6.61e-05
# Epoch 577/800: Loss=0.0240, LR=6.56e-05
# Epoch 578/800: Loss=0.0270, LR=6.50e-05
# Epoch 579/800: Loss=0.0234, LR=6.45e-05
# Epoch 580/800: Loss=0.0214, LR=6.40e-05
# Epoch 581/800: Loss=0.0252, LR=6.34e-05
# Epoch 580 Mini-IS: 5.19
# Epoch 582/800: Loss=0.0213, LR=6.29e-05
# Epoch 583/800: Loss=0.0228, LR=6.24e-05
# Epoch 584/800: Loss=0.0244, LR=6.18e-05
# Epoch 585/800: Loss=0.0234, LR=6.13e-05
# Epoch 586/800: Loss=0.0240, LR=6.08e-05
# Epoch 587/800: Loss=0.0228, LR=6.03e-05
# Epoch 588/800: Loss=0.0217, LR=5.97e-05
# Epoch 589/800: Loss=0.0217, LR=5.92e-05
# Epoch 590/800: Loss=0.0219, LR=5.87e-05
# Epoch 591/800: Loss=0.0252, LR=5.82e-05
# Epoch 590 Mini-IS: 5.06
# Epoch 592/800: Loss=0.0269, LR=5.77e-05
# Epoch 593/800: Loss=0.0246, LR=5.71e-05
# Epoch 594/800: Loss=0.0242, LR=5.66e-05
# Epoch 595/800: Loss=0.0207, LR=5.61e-05
# Epoch 596/800: Loss=0.0233, LR=5.56e-05
# Epoch 597/800: Loss=0.0233, LR=5.51e-05
# Epoch 598/800: Loss=0.0218, LR=5.46e-05
# Epoch 599/800: Loss=0.0239, LR=5.41e-05
# Epoch 600/800: Loss=0.0215, LR=5.36e-05
# Epoch 601/800: Loss=0.0208, LR=5.31e-05
# Epoch 600 Mini-IS: 5.03
# Epoch 602/800: Loss=0.0218, LR=5.26e-05
# Epoch 603/800: Loss=0.0205, LR=5.21e-05
# Epoch 604/800: Loss=0.0220, LR=5.16e-05
# Epoch 605/800: Loss=0.0231, LR=5.11e-05
# Epoch 606/800: Loss=0.0218, LR=5.06e-05
# Epoch 607/800: Loss=0.0249, LR=5.01e-05
# Epoch 608/800: Loss=0.0231, LR=4.96e-05
# Epoch 609/800: Loss=0.0264, LR=4.91e-05
# Epoch 610/800: Loss=0.0220, LR=4.87e-05
# Epoch 611/800: Loss=0.0261, LR=4.82e-05
# Epoch 610 Mini-IS: 5.28
# Epoch 612/800: Loss=0.0206, LR=4.77e-05
# Epoch 613/800: Loss=0.0217, LR=4.72e-05
# Epoch 614/800: Loss=0.0228, LR=4.67e-05
# Epoch 615/800: Loss=0.0235, LR=4.63e-05
# Epoch 616/800: Loss=0.0233, LR=4.58e-05
# Epoch 617/800: Loss=0.0206, LR=4.53e-05
# Epoch 618/800: Loss=0.0244, LR=4.49e-05
# Epoch 619/800: Loss=0.0226, LR=4.44e-05
# Epoch 620/800: Loss=0.0228, LR=4.39e-05
# Epoch 621/800: Loss=0.0215, LR=4.35e-05
# Epoch 620 Mini-IS: 5.10
# Epoch 622/800: Loss=0.0231, LR=4.30e-05
# Epoch 623/800: Loss=0.0194, LR=4.26e-05
# Epoch 624/800: Loss=0.0217, LR=4.21e-05
# Epoch 625/800: Loss=0.0220, LR=4.16e-05
# Epoch 626/800: Loss=0.0211, LR=4.12e-05
# Epoch 627/800: Loss=0.0225, LR=4.07e-05
# Epoch 628/800: Loss=0.0238, LR=4.03e-05
# Epoch 629/800: Loss=0.0232, LR=3.99e-05
# Epoch 630/800: Loss=0.0229, LR=3.94e-05
# Epoch 631/800: Loss=0.0242, LR=3.90e-05
# Epoch 630 Mini-IS: 5.10
# Epoch 632/800: Loss=0.0242, LR=3.85e-05
# Epoch 633/800: Loss=0.0222, LR=3.81e-05
# Epoch 634/800: Loss=0.0244, LR=3.77e-05
# Epoch 635/800: Loss=0.0230, LR=3.72e-05
# Epoch 636/800: Loss=0.0220, LR=3.68e-05
# Epoch 637/800: Loss=0.0199, LR=3.64e-05
# Epoch 638/800: Loss=0.0222, LR=3.59e-05
# Epoch 639/800: Loss=0.0235, LR=3.55e-05
# Epoch 640/800: Loss=0.0225, LR=3.51e-05
# Epoch 641/800: Loss=0.0232, LR=3.47e-05
# Epoch 640 Mini-IS: 5.23
# Epoch 642/800: Loss=0.0242, LR=3.43e-05
# Epoch 643/800: Loss=0.0220, LR=3.38e-05
# Epoch 644/800: Loss=0.0260, LR=3.34e-05
# Epoch 645/800: Loss=0.0212, LR=3.30e-05
# Epoch 646/800: Loss=0.0242, LR=3.26e-05
# Epoch 647/800: Loss=0.0230, LR=3.22e-05
# Epoch 648/800: Loss=0.0228, LR=3.18e-05
# Epoch 649/800: Loss=0.0203, LR=3.14e-05
# Epoch 650/800: Loss=0.0200, LR=3.10e-05
# Epoch 651/800: Loss=0.0210, LR=3.06e-05
# Epoch 650 Mini-IS: 5.23
# Epoch 652/800: Loss=0.0214, LR=3.02e-05
# Epoch 653/800: Loss=0.0229, LR=2.98e-05
# Epoch 654/800: Loss=0.0236, LR=2.94e-05
# Epoch 655/800: Loss=0.0246, LR=2.90e-05
# Epoch 656/800: Loss=0.0251, LR=2.86e-05
# Epoch 657/800: Loss=0.0221, LR=2.83e-05
# Epoch 658/800: Loss=0.0245, LR=2.79e-05
# Epoch 659/800: Loss=0.0202, LR=2.75e-05
# Epoch 660/800: Loss=0.0227, LR=2.71e-05
# Epoch 661/800: Loss=0.0232, LR=2.68e-05
# Epoch 660 Mini-IS: 5.56
# Epoch 662/800: Loss=0.0246, LR=2.64e-05
# Epoch 663/800: Loss=0.0233, LR=2.60e-05
# Epoch 664/800: Loss=0.0211, LR=2.56e-05
# Epoch 665/800: Loss=0.0214, LR=2.53e-05
# Epoch 666/800: Loss=0.0234, LR=2.49e-05
# Epoch 667/800: Loss=0.0222, LR=2.46e-05
# Epoch 668/800: Loss=0.0234, LR=2.42e-05
# Epoch 669/800: Loss=0.0227, LR=2.38e-05
# Epoch 670/800: Loss=0.0225, LR=2.35e-05
# Epoch 671/800: Loss=0.0223, LR=2.31e-05
# Epoch 670 Mini-IS: 5.75
# Epoch 672/800: Loss=0.0220, LR=2.28e-05
# Epoch 673/800: Loss=0.0239, LR=2.24e-05
# Epoch 674/800: Loss=0.0233, LR=2.21e-05
# Epoch 675/800: Loss=0.0224, LR=2.18e-05
# Epoch 676/800: Loss=0.0248, LR=2.14e-05
# Epoch 677/800: Loss=0.0225, LR=2.11e-05
# Epoch 678/800: Loss=0.0222, LR=2.08e-05
# Epoch 679/800: Loss=0.0233, LR=2.04e-05
# Epoch 680/800: Loss=0.0233, LR=2.01e-05
# Epoch 681/800: Loss=0.0271, LR=1.98e-05
# Epoch 680 Mini-IS: 5.39
# Epoch 682/800: Loss=0.0229, LR=1.94e-05
# Epoch 683/800: Loss=0.0243, LR=1.91e-05
# Epoch 684/800: Loss=0.0203, LR=1.88e-05
# Epoch 685/800: Loss=0.0200, LR=1.85e-05
# Epoch 686/800: Loss=0.0219, LR=1.82e-05
# Epoch 687/800: Loss=0.0210, LR=1.79e-05
# Epoch 688/800: Loss=0.0243, LR=1.76e-05
# Epoch 689/800: Loss=0.0246, LR=1.73e-05
# Epoch 690/800: Loss=0.0228, LR=1.69e-05
# Epoch 691/800: Loss=0.0216, LR=1.66e-05
# Epoch 690 Mini-IS: 5.18
# Epoch 692/800: Loss=0.0196, LR=1.63e-05
# Epoch 693/800: Loss=0.0208, LR=1.61e-05
# Epoch 694/800: Loss=0.0194, LR=1.58e-05
# Epoch 695/800: Loss=0.0199, LR=1.55e-05
# Epoch 696/800: Loss=0.0210, LR=1.52e-05
# Epoch 697/800: Loss=0.0229, LR=1.49e-05
# Epoch 698/800: Loss=0.0254, LR=1.46e-05
# Epoch 699/800: Loss=0.0211, LR=1.43e-05
# Epoch 700/800: Loss=0.0243, LR=1.41e-05
# Epoch 701/800: Loss=0.0215, LR=1.38e-05
# Epoch 700 Mini-IS: 4.94
# Epoch 702/800: Loss=0.0262, LR=1.35e-05
# Epoch 703/800: Loss=0.0227, LR=1.32e-05
# Epoch 704/800: Loss=0.0216, LR=1.30e-05
# Epoch 705/800: Loss=0.0240, LR=1.27e-05
# Epoch 706/800: Loss=0.0246, LR=1.24e-05
# Epoch 707/800: Loss=0.0249, LR=1.22e-05
# Epoch 708/800: Loss=0.0247, LR=1.19e-05
# Epoch 709/800: Loss=0.0261, LR=1.17e-05
# Epoch 710/800: Loss=0.0236, LR=1.14e-05
# Epoch 711/800: Loss=0.0226, LR=1.12e-05
# Epoch 710 Mini-IS: 4.91
# Epoch 712/800: Loss=0.0220, LR=1.09e-05
# Epoch 713/800: Loss=0.0246, LR=1.07e-05
# Epoch 714/800: Loss=0.0226, LR=1.04e-05
# Epoch 715/800: Loss=0.0224, LR=1.02e-05
# Epoch 716/800: Loss=0.0212, LR=9.96e-06
# Epoch 717/800: Loss=0.0215, LR=9.73e-06
# Epoch 718/800: Loss=0.0220, LR=9.50e-06
# Epoch 719/800: Loss=0.0239, LR=9.27e-06
# Epoch 720/800: Loss=0.0249, LR=9.05e-06
# Epoch 721/800: Loss=0.0239, LR=8.82e-06
# Epoch 720 Mini-IS: 5.46
# Epoch 722/800: Loss=0.0234, LR=8.60e-06
# Epoch 723/800: Loss=0.0209, LR=8.39e-06
# Epoch 724/800: Loss=0.0235, LR=8.17e-06
# Epoch 725/800: Loss=0.0227, LR=7.96e-06
# Epoch 726/800: Loss=0.0253, LR=7.75e-06
# Epoch 727/800: Loss=0.0208, LR=7.55e-06
# Epoch 728/800: Loss=0.0215, LR=7.34e-06
# Epoch 729/800: Loss=0.0217, LR=7.14e-06
# Epoch 730/800: Loss=0.0253, LR=6.94e-06
# Epoch 731/800: Loss=0.0233, LR=6.75e-06
# Epoch 730 Mini-IS: 5.43
# Epoch 732/800: Loss=0.0229, LR=6.55e-06
# Epoch 733/800: Loss=0.0224, LR=6.36e-06
# Epoch 734/800: Loss=0.0241, LR=6.18e-06
# Epoch 735/800: Loss=0.0226, LR=5.99e-06
# Epoch 736/800: Loss=0.0235, LR=5.81e-06
# Epoch 737/800: Loss=0.0238, LR=5.63e-06
# Epoch 738/800: Loss=0.0197, LR=5.46e-06
# Epoch 739/800: Loss=0.0242, LR=5.28e-06
# Epoch 740/800: Loss=0.0217, LR=5.11e-06
# Epoch 741/800: Loss=0.0222, LR=4.94e-06
# Epoch 740 Mini-IS: 5.38
# Epoch 742/800: Loss=0.0214, LR=4.78e-06
# Epoch 743/800: Loss=0.0218, LR=4.62e-06
# Epoch 744/800: Loss=0.0221, LR=4.46e-06
# Epoch 745/800: Loss=0.0241, LR=4.30e-06
# Epoch 746/800: Loss=0.0191, LR=4.15e-06
# Epoch 747/800: Loss=0.0192, LR=3.99e-06
# Epoch 748/800: Loss=0.0236, LR=3.85e-06
# Epoch 749/800: Loss=0.0221, LR=3.70e-06
# Epoch 750/800: Loss=0.0221, LR=3.56e-06
# Epoch 751/800: Loss=0.0250, LR=3.42e-06
# Epoch 750 Mini-IS: 5.18
# Epoch 752/800: Loss=0.0255, LR=3.28e-06
# Epoch 753/800: Loss=0.0228, LR=3.14e-06
# Epoch 754/800: Loss=0.0238, LR=3.01e-06
# Epoch 755/800: Loss=0.0215, LR=2.88e-06
# Epoch 756/800: Loss=0.0192, LR=2.76e-06
# Epoch 757/800: Loss=0.0224, LR=2.63e-06
# Epoch 758/800: Loss=0.0235, LR=2.51e-06
# Epoch 759/800: Loss=0.0195, LR=2.39e-06
# Epoch 760/800: Loss=0.0245, LR=2.28e-06
# Epoch 761/800: Loss=0.0246, LR=2.17e-06
# Epoch 760 Mini-IS: 5.15
# Epoch 762/800: Loss=0.0217, LR=2.06e-06
# Epoch 763/800: Loss=0.0227, LR=1.95e-06
# Epoch 764/800: Loss=0.0246, LR=1.85e-06
# Epoch 765/800: Loss=0.0219, LR=1.75e-06
# Epoch 766/800: Loss=0.0209, LR=1.65e-06
# Epoch 767/800: Loss=0.0218, LR=1.55e-06
# Epoch 768/800: Loss=0.0243, LR=1.46e-06
# Epoch 769/800: Loss=0.0199, LR=1.37e-06
# Epoch 770/800: Loss=0.0250, LR=1.28e-06
# Epoch 771/800: Loss=0.0240, LR=1.20e-06
# Epoch 770 Mini-IS: 5.26
# Epoch 772/800: Loss=0.0225, LR=1.12e-06
# Epoch 773/800: Loss=0.0231, LR=1.04e-06
# Epoch 774/800: Loss=0.0229, LR=9.65e-07
# Epoch 775/800: Loss=0.0238, LR=8.93e-07
# Epoch 776/800: Loss=0.0234, LR=8.23e-07
# Epoch 777/800: Loss=0.0192, LR=7.56e-07
# Epoch 778/800: Loss=0.0189, LR=6.92e-07
# Epoch 779/800: Loss=0.0235, LR=6.30e-07
# Epoch 780/800: Loss=0.0241, LR=5.72e-07
# Epoch 781/800: Loss=0.0217, LR=5.16e-07
# Epoch 780 Mini-IS: 4.97
# Epoch 782/800: Loss=0.0215, LR=4.63e-07
# Epoch 783/800: Loss=0.0243, LR=4.14e-07
# Epoch 784/800: Loss=0.0217, LR=3.66e-07
# Epoch 785/800: Loss=0.0218, LR=3.22e-07
# Epoch 786/800: Loss=0.0229, LR=2.81e-07
# Epoch 787/800: Loss=0.0255, LR=2.42e-07
# Epoch 788/800: Loss=0.0241, LR=2.07e-07
# Epoch 789/800: Loss=0.0231, LR=1.74e-07
# Epoch 790/800: Loss=0.0239, LR=1.44e-07
# Epoch 791/800: Loss=0.0244, LR=1.17e-07
# Epoch 790 Mini-IS: 5.28
# Epoch 792/800: Loss=0.0240, LR=9.25e-08
# Epoch 793/800: Loss=0.0220, LR=7.11e-08
# Epoch 794/800: Loss=0.0240, LR=5.26e-08
# Epoch 795/800: Loss=0.0241, LR=3.69e-08
# Epoch 796/800: Loss=0.0240, LR=2.40e-08
# Epoch 797/800: Loss=0.0248, LR=1.40e-08
# Epoch 798/800: Loss=0.0249, LR=6.90e-09
# Epoch 799/800: Loss=0.0236, LR=2.62e-09
# Epoch 800/800: Loss=0.0232, LR=1.20e-09

# --- Evaluating with 32 samples ---
# Total Generation Time: 0.7733 seconds
# FID Score: 262.9307
# Inception Score: 2.3671 ± 0.2544


# Total Generation Time: 1332.9670 seconds
# FID Score: 15.2991
# IS: 7.9147 ± 0.1056