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
# 1. Feature Hooks & Projector
# ==========================================
class FeatureHook:
    def __init__(self, module):
        self.features = None
        self.hook = module.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.features = output[0] if isinstance(output, tuple) else output
    def close(self):
        self.hook.remove()

class FeatureProjector(nn.Module):
    def __init__(self, student_channels=128, teacher_channels=256):
        super().__init__()
        self.proj = nn.Conv2d(student_channels, teacher_channels, kernel_size=1)
    def forward(self, x):
        return self.proj(x)

def get_predicted_x0(unet, scheduler, x_t, t, noise_pred):
    alphas_cumprod_device = scheduler.alphas_cumprod.to(x_t.device)
    alpha_prod_t = alphas_cumprod_device[t].view(-1, 1, 1, 1)
    beta_prod_t = 1 - alpha_prod_t
    return (x_t - torch.sqrt(beta_prod_t) * noise_pred) / torch.sqrt(alpha_prod_t)

# ==========================================
# 2. Evaluation Functions (50-Step & 5-Step)
# ==========================================
@torch.no_grad()
def generate_samples(unet, scheduler, batch_size, device, steps=50):
    """Generates images. Used for 50 steps in Phase 1, and 5 steps in Phase 2."""
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
        noise = torch.randn_like(x)
        x = torch.sqrt(alpha_prod_t_next) * pred_x0 + torch.sqrt(1 - alpha_prod_t_next) * noise

    return x.clamp(-1, 1)

@torch.no_grad()
def evaluate_student(student_unet, scheduler, dataloader, device, num_samples=2048, batch_size=64, steps=50):
    print(f"\n--- Evaluating on {num_samples} Samples ({steps} Steps) ---")
    student_unet.eval()
    fid_metric = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    is_metric = InceptionScore(normalize=True).to(device)
    
    real_processed = 0
    for imgs, _ in dataloader:
        if real_processed >= num_samples: break
        batch_imgs = imgs[:min(batch_size, num_samples - real_processed)].to(device)
        batch_imgs = (batch_imgs + 1.0) / 2.0 
        fid_metric.update(batch_imgs, real=True)
        real_processed += batch_imgs.shape[0]

    fake_processed = 0
    pbar = tqdm(total=num_samples, desc=f"Generating {steps}-Step Samples")
    while fake_processed < num_samples:
        current_batch_size = min(batch_size, num_samples - fake_processed)
        fake_imgs = generate_samples(student_unet, scheduler, current_batch_size, device, steps=steps)
        fake_imgs = (fake_imgs + 1.0) / 2.0 
        fid_metric.update(fake_imgs, real=False)
        is_metric.update(fake_imgs)
        fake_processed += current_batch_size
        pbar.update(current_batch_size)
    
    fid_score = fid_metric.compute().item()
    is_score_mean, is_score_std = is_metric.compute()
    print(f"Results -> FID: {fid_score:.4f} | IS: {is_score_mean.item():.4f} ± {is_score_std.item():.4f}")
    
    fid_metric.reset()
    is_metric.reset()
    student_unet.train()
    return fid_score

# ==========================================
# 3. Main Two-Phase Loop
# ==========================================
def run_two_phase_distillation():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    print("Loading Teacher Model...")
    teacher_pipe = DDPMPipeline.from_pretrained("google/ddpm-cifar10-32").to(device)
    teacher_unet = teacher_pipe.unet
    teacher_unet.eval() 
    for param in teacher_unet.parameters(): param.requires_grad = False
    
    scheduler = DDIMScheduler.from_config(teacher_pipe.scheduler.config)
    scheduler.set_timesteps(50) 
    timesteps = scheduler.timesteps.to(device)

    print("Initializing Lightweight Student...")
    student_config = teacher_unet.config.copy()
    student_config["block_out_channels"] = (64, 128, 128, 128) 
    student_unet = UNet2DModel.from_config(student_config).to(device)
    student_unet.train()

    projector = FeatureProjector(student_channels=128, teacher_channels=256).to(device)
    projector.train()

    teacher_hook = FeatureHook(teacher_unet.mid_block)
    student_hook = FeatureHook(student_unet.mid_block)

    BATCH_SIZE = 64
    PHASE_1_EPOCHS = 50 # Teach it to draw
    PHASE_2_EPOCHS = 50 # Teach it to be fast (50 total)
    LR = 1e-4

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)

    optimizer = torch.optim.AdamW(list(student_unet.parameters()) + list(projector.parameters()), lr=LR)

    # evaluate_student(student_unet, scheduler, dataloader, device, num_samples=50_000, batch_size=64, steps=5)

    # ==========================================
    # PHASE 1: Feature Pre-Training (Learn to Draw)
    # ==========================================
    print("\n" + "="*40)
    print("PHASE 1: STANDARD KNOWLEDGE DISTILLATION")
    print("="*40)
    
    for epoch in range(PHASE_1_EPOCHS):
        pbar = tqdm(dataloader, desc=f"Phase 1 - Epoch {epoch+1}/{PHASE_1_EPOCHS}")
        for imgs, _ in pbar:
            imgs = imgs.to(device)
            B = imgs.shape[0]
            
            # Pick a random timestep
            t = torch.randint(0, len(timesteps), (B,), device=device)
            actual_t = timesteps[t]

            alphas_cumprod_device = scheduler.alphas_cumprod.to(device)
            noise = torch.randn_like(imgs)
            alphas_cumprod_t = alphas_cumprod_device[actual_t].view(-1, 1, 1, 1)
            x_t = torch.sqrt(alphas_cumprod_t) * imgs + torch.sqrt(1 - alphas_cumprod_t) * noise

            # 1. Teacher predicts noise (and fills feature hook)
            with torch.no_grad():
                teacher_noise_pred = teacher_unet(x_t, actual_t).sample

            # 2. Student predicts noise (and fills feature hook)
            student_noise_pred = student_unet(x_t, actual_t).sample

            # 3. Phase 1 Losses (Mimic the teacher exactly)
            loss_noise = F.mse_loss(student_noise_pred, teacher_noise_pred)
            loss_feature = F.mse_loss(projector(student_hook.features), teacher_hook.features)
            
            loss = loss_noise + (0.5 * loss_feature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix({'Noise': f"{loss_noise.item():.3f}", 'Feat': f"{loss_feature.item():.3f}"})

        # Evaluate Phase 1 using 50 STEPS (Is it learning to draw?)
        if (epoch + 1) % 5 == 0:
            evaluate_student(student_unet, scheduler, dataloader, device, num_samples=2048, batch_size=BATCH_SIZE, steps=50)

    evaluate_student(student_unet, scheduler, dataloader, device, num_samples=10_000, batch_size=BATCH_SIZE, steps=50)
    print("\nPhase 1 Complete! The Student should now know how to draw in 20 steps.")
    torch.save(student_unet.state_dict(), "student_unet_phase1_complete.pt")

    # ==========================================
    # PHASE 2: Consistency Distillation (Learn to be Fast)
    # ==========================================
    print("\n" + "="*40)
    print("PHASE 2: CONSISTENCY DISTILLATION")
    print("="*40)
    
    # Initialize Target Network now that Phase 1 is done
    target_unet = copy.deepcopy(student_unet)
    target_unet.eval()
    for param in target_unet.parameters(): param.requires_grad = False
    
    EMA_RATE = 0.95

    for epoch in range(PHASE_2_EPOCHS):
        pbar = tqdm(dataloader, desc=f"Phase 2 - Epoch {epoch+1}/{PHASE_2_EPOCHS}")
        for imgs, _ in pbar:
            imgs = imgs.to(device)
            B = imgs.shape[0]

            # In Phase 2, we need adjacent steps t_n and t_{n+1}
            step_indices = torch.randint(0, len(timesteps) - 1, (B,), device=device)
            t_n_plus_1 = timesteps[step_indices]
            t_n = timesteps[step_indices + 1]

            alphas_cumprod_device = scheduler.alphas_cumprod.to(device)
            noise = torch.randn_like(imgs)
            alphas_cumprod_n_plus_1 = alphas_cumprod_device[t_n_plus_1].view(-1, 1, 1, 1)
            x_t_n_plus_1 = torch.sqrt(alphas_cumprod_n_plus_1) * imgs + torch.sqrt(1 - alphas_cumprod_n_plus_1) * noise

            # 1. Teacher simulates 1 step backwards
            with torch.no_grad():
                teacher_noise_pred = teacher_unet(x_t_n_plus_1, t_n_plus_1).sample
                alpha_prod_t = alphas_cumprod_device[t_n_plus_1].view(-1, 1, 1, 1)
                alpha_prod_t_prev = alphas_cumprod_device[t_n].view(-1, 1, 1, 1)
                
                pred_x0_teacher = (x_t_n_plus_1 - torch.sqrt(1 - alpha_prod_t) * teacher_noise_pred) / torch.sqrt(alpha_prod_t)
                dir_xt_teacher = torch.sqrt(1 - alpha_prod_t_prev) * teacher_noise_pred
                x_t_n = torch.sqrt(alpha_prod_t_prev) * pred_x0_teacher + dir_xt_teacher

            # 2. Student predicts x0
            student_noise_pred = student_unet(x_t_n_plus_1, t_n_plus_1).sample
            student_pred_x0 = get_predicted_x0(student_unet, scheduler, x_t_n_plus_1, t_n_plus_1, student_noise_pred)

            # 3. Target predicts x0
            with torch.no_grad():
                target_noise_pred = target_unet(x_t_n, t_n).sample
                target_pred_x0 = get_predicted_x0(target_unet, scheduler, x_t_n, t_n, target_noise_pred)

            # 4. Phase 2 Losses (Consistency + Feature)
            loss_consistency = F.smooth_l1_loss(student_pred_x0, target_pred_x0)
            loss_feature = F.mse_loss(projector(student_hook.features), teacher_hook.features)
            loss = loss_consistency + (0.1 * loss_feature) # Lower feature weight in Phase 2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 5. Update Target Network
            with torch.no_grad():
                for param_student, param_target in zip(student_unet.parameters(), target_unet.parameters()):
                    param_target.data.mul_(EMA_RATE).add_(param_student.data, alpha=1 - EMA_RATE)

            pbar.set_postfix({'Consist': f"{loss_consistency.item():.3f}", 'Feat': f"{loss_feature.item():.3f}"})

        # Evaluate Phase 2 using 5 STEPS (Is it learning to be fast?)
        if (epoch + 1) % 5 == 0:
            evaluate_student(student_unet, scheduler, dataloader, device, num_samples=2048, batch_size=BATCH_SIZE, steps=5)

    teacher_hook.close()
    student_hook.close()
    torch.save(student_unet.state_dict(), "student_unet_phase2_final.pt")
    print("Training Complete!")

if __name__ == "__main__":
    run_two_phase_distillation()


# ========================================
# PHASE 1: STANDARD KNOWLEDGE DISTILLATION
# ========================================
# Phase 1 - Epoch 1/20: 100%|█████████████████████████████████████████████████████████| 781/781 [04:29<00:00,  2.89it/s, Noise=0.055, Feat=114.407]
# Phase 1 - Epoch 2/20: 100%|██████████████████████████████████████████████████████████| 781/781 [04:28<00:00,  2.91it/s, Noise=0.020, Feat=80.321]
# Phase 1 - Epoch 3/20: 100%|██████████████████████████████████████████████████████████| 781/781 [04:28<00:00,  2.91it/s, Noise=0.018, Feat=85.367]
# Phase 1 - Epoch 4/20: 100%|██████████████████████████████████████████████████████████| 781/781 [04:26<00:00,  2.93it/s, Noise=0.012, Feat=73.178]
# Phase 1 - Epoch 5/20: 100%|██████████████████████████████████████████████████████████| 781/781 [04:26<00:00,  2.93it/s, Noise=0.015, Feat=84.309]

# --- Evaluating on 2048 Samples (50 Steps) ---
# C:\Users\Dimitar Trajkov\anaconda3\Lib\site-packages\torchmetrics\utilities\prints.py:43: UserWarning: Metric `InceptionScore` will save all extracted features in buffer. For large datasets this may lead to large memory footprint.
#   warnings.warn(*args, **kwargs)
# Generating 50-Step Samples: 100%|████████████████████████████████████████████████████████████████████████████| 2048/2048 [01:41<00:00, 19.76it/s]Results -> FID: 174.8060 | IS: 2.6579 ± 0.0648
# Generating 50-Step Samples: 100%|████████████████████████████████████████████████████████████████████████████| 2048/2048 [01:46<00:00, 19.31it/s]
# Phase 1 - Epoch 6/20: 100%|██████████████████████████████████████████████████████████| 781/781 [04:26<00:00,  2.93it/s, Noise=0.011, Feat=71.402]
# Phase 1 - Epoch 7/20: 100%|██████████████████████████████████████████████████████████| 781/781 [04:24<00:00,  2.95it/s, Noise=0.009, Feat=68.323]
# Phase 1 - Epoch 8/20: 100%|██████████████████████████████████████████████████████████| 781/781 [04:24<00:00,  2.95it/s, Noise=0.005, Feat=58.065]
# Phase 1 - Epoch 9/20: 100%|██████████████████████████████████████████████████████████| 781/781 [04:26<00:00,  2.93it/s, Noise=0.012, Feat=67.412]
# Phase 1 - Epoch 10/20: 100%|█████████████████████████████████████████████████████████| 781/781 [04:25<00:00,  2.94it/s, Noise=0.006, Feat=65.161]

# --- Evaluating on 2048 Samples (50 Steps) ---
# Generating 50-Step Samples: 100%|████████████████████████████████████████████████████████████████████████████| 2048/2048 [01:39<00:00, 21.38it/s]Results -> FID: 175.4747 | IS: 3.2202 ± 0.1434
# Generating 50-Step Samples: 100%|████████████████████████████████████████████████████████████████████████████| 2048/2048 [01:42<00:00, 19.97it/s]
# Phase 1 - Epoch 11/20: 100%|█████████████████████████████████████████████████████████| 781/781 [04:26<00:00,  2.93it/s, Noise=0.007, Feat=47.801]
# Phase 1 - Epoch 12/20: 100%|█████████████████████████████████████████████████████████| 781/781 [04:29<00:00,  2.90it/s, Noise=0.006, Feat=65.823]
# Phase 1 - Epoch 13/20: 100%|█████████████████████████████████████████████████████████| 781/781 [04:29<00:00,  2.90it/s, Noise=0.009, Feat=62.304]
# Phase 1 - Epoch 14/20: 100%|█████████████████████████████████████████████████████████| 781/781 [04:27<00:00,  2.92it/s, Noise=0.003, Feat=55.952]
# Phase 1 - Epoch 15/20: 100%|█████████████████████████████████████████████████████████| 781/781 [04:27<00:00,  2.92it/s, Noise=0.006, Feat=63.529]

# --- Evaluating on 2048 Samples (50 Steps) ---
# Generating 50-Step Samples: 100%|████████████████████████████████████████████████████████████████████████████| 2048/2048 [01:38<00:00, 20.83it/s]Results -> FID: 181.0850 | IS: 2.7863 ± 0.1303
# Generating 50-Step Samples: 100%|████████████████████████████████████████████████████████████████████████████| 2048/2048 [01:42<00:00, 20.07it/s]
# Phase 1 - Epoch 16/20: 100%|█████████████████████████████████████████████████████████| 781/781 [04:29<00:00,  2.90it/s, Noise=0.007, Feat=64.899]
# Phase 1 - Epoch 17/20: 100%|█████████████████████████████████████████████████████████| 781/781 [04:28<00:00,  2.91it/s, Noise=0.007, Feat=63.392]
# Phase 1 - Epoch 18/20: 100%|█████████████████████████████████████████████████████████| 781/781 [04:28<00:00,  2.91it/s, Noise=0.011, Feat=73.629]
# Phase 1 - Epoch 19/20: 100%|█████████████████████████████████████████████████████████| 781/781 [04:27<00:00,  2.92it/s, Noise=0.011, Feat=74.076]
# Phase 1 - Epoch 20/20: 100%|█████████████████████████████████████████████████████████| 781/781 [04:28<00:00,  2.90it/s, Noise=0.006, Feat=63.053]

# --- Evaluating on 2048 Samples (50 Steps) ---
# Generating 50-Step Samples: 100%|████████████████████████████████████████████████████████████████████████████| 2048/2048 [01:39<00:00, 21.36it/s]Results -> FID: 175.1943 | IS: 2.7886 ± 0.0938
# Generating 50-Step Samples: 100%|████████████████████████████████████████████████████████████████████████████| 2048/2048 [01:43<00:00, 19.85it/s]

# --- Evaluating on 10000 Samples (50 Steps) ---
# Generating 50-Step Samples: 100%|████████████████████████████████████████████████████████████████████████| 10000/10000 [1:14:20<00:00, 17.57it/s]Results -> FID: 161.1936 | IS: 2.8099 ± 0.0451
# Generating 50-Step Samples: 100%|████████████████████████████████████████████████████████████████████████| 10000/10000 [1:14:23<00:00,  2.24it/s]

# Phase 1 Complete! The Student should now know how to draw in 50 steps.
