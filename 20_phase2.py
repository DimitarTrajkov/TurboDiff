import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from diffusers import DDPMPipeline, DDIMScheduler, UNet2DModel
import copy
from tqdm import tqdm
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore

# ==========================================
# 1. Helpers & Hooks
# ==========================================
class FeatureHook:
    def __init__(self, module):
        self.features = None
        self.hook = module.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.features = output[0] if isinstance(output, tuple) else output
    def close(self):
        self.hook.remove()

def get_predicted_x0(unet, scheduler, x_t, t, noise_pred):
    alphas_cumprod_device = scheduler.alphas_cumprod.to(x_t.device)
    alpha_prod_t = alphas_cumprod_device[t].view(-1, 1, 1, 1)
    beta_prod_t = 1 - alpha_prod_t
    return (x_t - torch.sqrt(beta_prod_t) * noise_pred) / torch.sqrt(alpha_prod_t)

@torch.no_grad()
def generate_samples(unet, scheduler, batch_size, device, steps=50):
    """Deterministic DDIM Sampling"""
    timesteps = torch.linspace(scheduler.config.num_train_timesteps - 1, 0, steps, dtype=torch.long, device=device)
    x = torch.randn(batch_size, 3, 32, 32, device=device)
    alphas_cumprod_device = scheduler.alphas_cumprod.to(device)
    for i, t in enumerate(timesteps):
        t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
        noise_pred = unet(x, t_batch).sample
        alpha_prod_t = alphas_cumprod_device[t_batch].view(-1, 1, 1, 1)
        pred_x0 = (x - torch.sqrt(1 - alpha_prod_t) * noise_pred) / torch.sqrt(alpha_prod_t)
        if i == len(timesteps) - 1:
            x = pred_x0
            break
        next_t = timesteps[i + 1]
        next_t_batch = torch.full((batch_size,), next_t, device=device, dtype=torch.long)
        alpha_prod_t_next = alphas_cumprod_device[next_t_batch].view(-1, 1, 1, 1)
        x = torch.sqrt(alpha_prod_t_next) * pred_x0 + torch.sqrt(1 - alpha_prod_t_next) * noise_pred
    return x.clamp(-1, 1)

@torch.no_grad()
def evaluate_student(student_unet, scheduler, dataloader, device, num_samples=2048, batch_size=64, steps=1):
    """Evaluation supporting 1-step or multi-step"""
    student_unet.eval()
    fid_metric = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    is_metric = InceptionScore(normalize=True).to(device)
    
    # Real images
    real_processed = 0
    for imgs, _ in dataloader:
        if real_processed >= num_samples: break
        batch_imgs = (imgs[:min(batch_size, num_samples - real_processed)].to(device) + 1.0) / 2.0 
        fid_metric.update(batch_imgs, real=True)
        real_processed += batch_imgs.shape[0]

    # Fake images
    fake_processed = 0
    pbar = tqdm(total=num_samples, desc=f"Evaluating {steps}-Step")
    while fake_processed < num_samples:
        current_batch_size = min(batch_size, num_samples - fake_processed)
        fake_imgs = generate_samples(student_unet, scheduler, current_batch_size, device, steps=steps)
        fake_imgs = (fake_imgs + 1.0) / 2.0 
        fid_metric.update(fake_imgs, real=False)
        is_metric.update(fake_imgs)
        fake_processed += current_batch_size
        pbar.update(current_batch_size)
    
    fid_score = fid_metric.compute().item()
    is_mean, _ = is_metric.compute()
    print(f"Results ({steps} steps) -> FID: {fid_score:.4f} | IS: {is_mean.item():.4f}")
    student_unet.train()
    return fid_score

# ==========================================
# 2. Main Training Function
# ==========================================
def train_phase_2():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # --- 1. Load Models ---
    teacher_pipe = DDPMPipeline.from_pretrained("google/ddpm-cifar10-32").to(device)
    teacher_unet = teacher_pipe.unet
    teacher_unet.eval()
    
    scheduler = DDIMScheduler.from_config(teacher_pipe.scheduler.config)
    scheduler.set_timesteps(50) # Consistency steps based on 50 discrete points
    timesteps = scheduler.timesteps.to(device)

    student_config = teacher_unet.config.copy()
    student_config["block_out_channels"] = (64, 128, 128, 128) 
    student_unet = UNet2DModel.from_config(student_config).to(device)

    # LOAD PHASE 1 WEIGHTS
    print("Loading Phase 1 Checkpoint...")
    student_unet.load_state_dict(torch.load("student_unet_phase1_complete.pt", map_location=device, weights_only=True))

    # EMA Target Model (Initialized from Student)
    target_unet = copy.deepcopy(student_unet).to(device)
    target_unet.eval()
    for param in target_unet.parameters(): param.requires_grad = False

    # --- 2. Params (REPAIRED) ---
    BATCH_SIZE = 128 
    PHASE_2_EPOCHS = 30
    LR = 5e-5           # Lower LR for stability
    EMA_RATE = 0.999    # Much higher EMA for stability
    
    optimizer = torch.optim.AdamW(student_unet.parameters(), lr=LR)
    
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    # --- 3. Phase 2 Loop ---
    print("\n" + "="*40)
    print("PHASE 2: CONSISTENCY DISTILLATION (STABILIZED)")
    print("="*40)

    for epoch in range(PHASE_2_EPOCHS):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{PHASE_2_EPOCHS}")
        for imgs, _ in pbar:
            imgs = imgs.to(device)
            B = imgs.shape[0]

            # Pick timestep pairs (t_n+1, t_n)
            indices = torch.randint(0, len(timesteps) - 1, (B,), device=device)
            t_next = timesteps[indices]
            t_curr = timesteps[indices + 1]

            # Prepare noisy x_{t_next}
            noise = torch.randn_like(imgs)
            alphas = scheduler.alphas_cumprod.to(device)
            x_next = torch.sqrt(alphas[t_next]).view(-1,1,1,1) * imgs + torch.sqrt(1 - alphas[t_next]).view(-1,1,1,1) * noise

            # 1. Get Teacher's ground truth for the jump
            with torch.no_grad():
                t_noise_pred = teacher_unet(x_next, t_next).sample
                # DDIM Step to get slightly cleaner x_curr
                alpha_next = alphas[t_next].view(-1,1,1,1)
                alpha_curr = alphas[t_curr].view(-1,1,1,1)
                pred_x0_t = (x_next - torch.sqrt(1 - alpha_next) * t_noise_pred) / torch.sqrt(alpha_next)
                x_curr = torch.sqrt(alpha_curr) * pred_x0_t + torch.sqrt(1 - alpha_curr) * t_noise_pred

            # 2. Student predicts x0 from x_next
            s_noise_pred = student_unet(x_next, t_next).sample
            s_x0 = get_predicted_x0(student_unet, scheduler, x_next, t_next, s_noise_pred)

            # 3. Target (EMA) predicts x0 from x_curr
            with torch.no_grad():
                target_noise_pred = target_unet(x_curr, t_curr).sample
                target_x0 = get_predicted_x0(target_unet, scheduler, x_curr, t_curr, target_noise_pred)

            # Consistency Loss
            loss = F.mse_loss(s_x0, target_x0)

            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping is very helpful for Phase 2
            torch.nn.utils.clip_grad_norm_(student_unet.parameters(), 1.0)
            optimizer.step()

            # Update EMA Target
            with torch.no_grad():
                for p, pt in zip(student_unet.parameters(), target_unet.parameters()):
                    pt.data.mul_(EMA_RATE).add_(p.data, alpha=1 - EMA_RATE)

            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        if (epoch + 1) % 5 == 0:
            # Test 1-step generation
            evaluate_student(student_unet, scheduler, dataloader, device, num_samples=2048, steps=1)

    torch.save(student_unet.state_dict(), "student_unet_phase2_final.pt")
    print("Training Complete!")

if __name__ == "__main__":
    train_phase_2()