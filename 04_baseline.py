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
# ==========================================
# 1. Fourier Domain Utilities (CFT/FFT)
# ==========================================
def img_to_fft(x):
    """Converts (B, 3, H, W) to (B, 6, H, W) using 2D FFT."""
    # Orthogonal norm ensures energy is preserved
    fft_c = torch.fft.fft2(x, norm='ortho')
    # Concatenate real and imaginary parts as channels
    return torch.cat([fft_c.real, fft_c.imag], dim=1)

def fft_to_img(z):
    """Converts (B, 6, H, W) back to (B, 3, H, W) image."""
    real, imag = z.chunk(2, dim=1)
    fft_c = torch.complex(real, imag)
    img = torch.fft.ifft2(fft_c, norm='ortho').real
    return torch.clamp(img, -1.0, 1.0)

# ==========================================
# 2. Architecture Components
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
    """O(N) Attention using ELU feature maps."""
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

        # Apply feature map to make values positive
        q = F.elu(q) + 1.0
        k = F.elu(k) + 1.0

        # Linear Attention: q * (k^T * v)
        kv = torch.matmul(k.transpose(-2, -1), v) # (B, Heads, Head_Dim, Head_Dim)
        out = torch.matmul(q, kv) # (B, Heads, N, Head_Dim)
        
        # Normalization denominator
        k_sum = k.sum(dim=-2) 
        denom = torch.matmul(q, k_sum.unsqueeze(-1)).squeeze(-1) + 1e-6
        out = out / denom.unsqueeze(-1)
        
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.proj(out)

class SwiGLU(nn.Module):
    """Swish-Gated Linear Unit for MLP block."""
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
    def __init__(self, img_size=32, patch_size=4, in_channels=6, dim=256, depth=6, num_heads=8):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_patches = (img_size // patch_size) ** 2
        
        self.x_embedder = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)
        self.t_embedder = TimestepEmbedder(dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, dim))
        
        self.blocks = nn.ModuleList([DiTBlock(dim, num_heads) for _ in range(depth)])
        self.norm_final = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim, bias=True))
        self.proj_out = nn.Linear(dim, patch_size * patch_size * in_channels)

    def unpatchify(self, x):
        """Convert sequence back to image spatial dimensions."""
        B, N, _ = x.shape
        p = self.patch_size
        h = w = int(math.sqrt(N))
        x = x.reshape(B, h, w, self.in_channels, p, p)
        x = torch.einsum('nhwcpq->nchpwq', x)
        return x.reshape(B, self.in_channels, h * p, w * p)

    def forward(self, x, t):
        # x: (B, 6, 32, 32) -> Patchify
        x = self.x_embedder(x).flatten(2).transpose(1, 2) # (B, N, D)
        x = x + self.pos_embed
        c = self.t_embedder(t)
        
        for block in self.blocks:
            x = block(x, c)
            
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = self.norm_final(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        x = self.proj_out(x)
        return self.unpatchify(x)

# ==========================================
# 3. Sampling & Evaluation
# ==========================================
@torch.no_grad()
def sample_4_steps(model, batch_size, device, steps=4):
    model.eval()
    # Sample pure noise in FFT domain
    x_t = torch.randn(batch_size, 6, 32, 32, device=device)
    dt = 1.0 / steps
    
    start_time = time.time()
    for i in range(steps):
        t = torch.full((batch_size,), i * dt, device=device)
        v_pred = model(x_t, t)
        x_t = x_t + v_pred * dt # Euler step
        
    generation_time = time.time() - start_time
    
    # Convert back to pixel domain
    images = fft_to_img(x_t)
    model.train()
    return images, generation_time

def evaluate_metrics(model, dataloader, device, num_samples=32):
    print(f"\n--- Evaluating with {num_samples} samples ---")
    fid = FrechetInceptionDistance(feature=64).to(device)
    inc = InceptionScore().to(device)

    # 1. Get Real Images for FID
    real_imgs = []
    for batch, _ in dataloader:
        real_imgs.append(batch)
        if sum(b.shape[0] for b in real_imgs) >= num_samples:
            break
    real_imgs = torch.cat(real_imgs, dim=0)[:num_samples].to(device)
    
    # Convert real [-1, 1] to [0, 255] uint8
    real_imgs_uint8 = ((real_imgs + 1.0) * 127.5).clamp(0, 255).byte()
    fid.update(real_imgs_uint8, real=True)

    # 2. Generate Fake Images
    fake_imgs, gen_time = sample_4_steps(model, num_samples, device, steps=4)
    fake_imgs_uint8 = ((fake_imgs + 1.0) * 127.5).clamp(0, 255).byte()
    
    fid.update(fake_imgs_uint8, real=False)
    inc.update(fake_imgs_uint8)

    fid_score = fid.compute().item()
    is_score_mean, is_score_std = inc.compute()
    
    print(f"Generation Time ({num_samples} images, 4 steps): {gen_time:.4f} seconds")
    print(f"FID Score: {fid_score:.4f}")
    print(f"Inception Score: {is_score_mean.item():.4f} ± {is_score_std.item():.4f}")
    
    grid = make_grid(fake_imgs[:32], nrow=8, normalize=True, value_range=(-1, 1))
    
    # Convert the PyTorch tensor to a NumPy array and change channel order for plotting (C, H, W) -> (H, W, C)
    grid_np = grid.permute(1, 2, 0).cpu().numpy()
    
    # Plot using matplotlib
    plt.imshow(grid_np)
    plt.title(f"Generated Samples (FID: {fid_score:.2f} | IS: {is_score_mean.item():.2f})")
    plt.axis('off')
    
    # Save the plot to the current directory
    plt.savefig("generated_samples.png", bbox_inches='tight', dpi=150)
    
    # Clear the current plot to free up memory for the next epoch
    plt.clf() 
    plt.close()
    
    print("-> Saved visual sample grid to 'generated_samples.png'")
    
# ==========================================
# 4. Main Training Loop (Rectified Flow)
# ==========================================
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Hyperparameters
    BATCH_SIZE = 256
    EPOCHS = 3 # Set higher (e.g. 200+) for actual convergence
    LR = 3e-4

    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Map to [-1, 1]
    ])
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True)

    # Init Model & Optimizer
    model = FastDiT(img_size=32, patch_size=4, in_channels=6, dim=256, depth=8, num_heads=8).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")

    # Training Loop
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for imgs, _ in pbar:
            imgs = imgs.to(device)
            B = imgs.shape[0]

            # 1. Transform to CFT (Frequency) Latent Space
            x_1 = img_to_fft(imgs) 
            
            # 2. Flow Matching Objective
            x_0 = torch.randn_like(x_1) # Pure Gaussian noise in frequency domain
            t = torch.rand(B, 1, 1, 1, device=device) # Sample random timestep
            
            # Linear trajectory: x_t = t * x_1 + (1 - t) * x_0
            x_t = t * x_1 + (1.0 - t) * x_0
            
            # Target vector field: v = x_1 - x_0
            target_v = x_1 - x_0
            
            # 3. Predict & Optimize
            optimizer.zero_grad()
            pred_v = model(x_t, t.squeeze())
            
            loss = F.mse_loss(pred_v, target_v)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'Loss': f"{loss.item():.4f}"})

        # Evaluate at the end of epoch
        if epoch % 10 == 0:
            evaluate_metrics(model, dataloader, device, num_samples=32)
            
    # Save final model
    evaluate_metrics(model, dataloader, device, num_samples=32)
    torch.save(model.state_dict(), "fast_dit_cifar10_3ep.pt")