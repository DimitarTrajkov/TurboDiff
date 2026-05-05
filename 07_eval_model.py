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

@torch.no_grad()
def evaluate_metrics(
    model,
    dataloader,
    device,
    num_samples=10000,
    batch_size=128,   # <-- NEW (generation batch size)
    steps=30,
    epoch=None
):
    print(f"\n--- Evaluating with {num_samples} samples (batch={batch_size}) ---")

    fid = FrechetInceptionDistance(feature=64).to(device)
    inc = InceptionScore().to(device)

    # =========================
    # 1. REAL IMAGES (streamed)
    # =========================
    seen = 0
    for imgs, _ in dataloader:
        imgs = imgs.to(device)

        b = imgs.shape[0]
        if seen + b > num_samples:
            imgs = imgs[:num_samples - seen]

        imgs_uint8 = ((imgs + 1.0) * 127.5).clamp(0, 255).byte()
        fid.update(imgs_uint8, real=True)

        seen += imgs.shape[0]
        if seen >= num_samples:
            break

    # =========================
    # 2. FAKE IMAGES (chunked)
    # =========================
    generated = 0
    start_time = time.time()

    while generated < num_samples:
        current_bs = min(batch_size, num_samples - generated)

        fake_imgs, _ = sample_euler(model, current_bs, device, steps=steps)
        fake_imgs_uint8 = ((fake_imgs + 1.0) * 127.5).clamp(0, 255).byte()

        fid.update(fake_imgs_uint8, real=False)
        inc.update(fake_imgs_uint8)

        generated += current_bs

    generation_time = time.time() - start_time

    # =========================
    # 3. METRICS
    # =========================
    fid_score = fid.compute().item()
    is_mean, is_std = inc.compute()

    print(f"Generation Time: {generation_time:.2f}s")
    print(f"FID: {fid_score:.4f}")
    print(f"IS: {is_mean.item():.4f} ± {is_std.item():.4f}")

    # =========================
    # 4. VISUALIZATION (only small batch)
    # =========================
    fake_imgs_vis, _ = sample_euler(model, 32, device, steps=steps)

    grid = make_grid(fake_imgs_vis, nrow=8, normalize=True, value_range=(-1, 1))
    grid_np = grid.permute(1, 2, 0).cpu().numpy()

    plt.imshow(grid_np)
    plt.title(f"FID: {fid_score:.2f} | IS: {is_mean.item():.2f}")
    plt.axis('off')
    plt.savefig(f"generated_samples_{epoch}.png", bbox_inches='tight', dpi=150)
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
    
    real_imgs_uint8 = ((real_imgs + 1.0) * 127.5).clamp(0, 255).byte()

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

    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Recreate the model architecture EXACTLY the same
    model = FastDiT(img_size=32, patch_size=4, in_channels=3, dim=256, depth=8, num_heads=8)
    # 2. Wrap with EMA container (same as training)
    ema_model = AveragedModel(model)
    # 3. Load weights
    ema_model.load_state_dict(torch.load("fast_dit_cifar10_epoch161_best.pt", map_location=device))

    # 4. Move to device and eval mode
    ema_model = ema_model.to(device)
    ema_model.eval()

    # Evaluate using the EMA model for much better results
    evaluate_metrics(
        ema_model,
        dataloader,
        device,
        num_samples=10000,
        batch_size=512,   # tune this to your GPU
        steps=30,
        epoch="big_eval"
    )