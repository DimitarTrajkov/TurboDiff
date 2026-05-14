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
# 1. Feature Hooks
# ==========================================
class FeatureHook:
    def __init__(self, module):
        self.features = None
        self.hook = module.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        # Handle different output types from Diffusers blocks
        self.features = output[0] if isinstance(output, tuple) else output
    def close(self):
        self.hook.remove()

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
    """Generates images using deterministic DDIM sampling."""
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
        
        # FIX 1: Use noise_pred instead of random noise for deterministic DDIM step
        x = torch.sqrt(alpha_prod_t_next) * pred_x0 + torch.sqrt(1 - alpha_prod_t_next) * noise_pred

    return x.clamp(-1, 1)

@torch.no_grad()
def generate_1_step_samples(unet, scheduler, batch_size, device):
    """One-shot generation for Consistency Models."""
    unet.eval()
    x = torch.randn(batch_size, 3, 32, 32, device=device)
    t_start = torch.full((batch_size,), scheduler.config.num_train_timesteps - 1, device=device, dtype=torch.long)
    noise_pred = unet(x, t_start).sample
    x_0 = get_predicted_x0(unet, scheduler, x, t_start, noise_pred)

    return x_0.clamp(-1, 1)

@torch.no_grad()
def generate_multistep_consistency(unet, scheduler, batch_size, device, steps=5):
    unet.eval()
    # 1. Start with pure noise
    x = torch.randn(batch_size, 3, 32, 32, device=device)
    
    # 2. Create 5 evenly spaced timesteps from 999 down to 0
    timesteps = torch.linspace(scheduler.config.num_train_timesteps - 1, 0, steps, dtype=torch.long, device=device)
    
    for i in range(len(timesteps)):
        t = timesteps[i]
        t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
        noise_pred = unet(x, t_batch).sample
        x_0 = get_predicted_x0(unet, scheduler, x, t_batch, noise_pred)
        
        if i < len(timesteps) - 1:
            next_t = timesteps[i + 1]
            next_t_batch = torch.full((batch_size,), next_t, device=device, dtype=torch.long)
            noise = torch.randn_like(x)
            alpha_prod_t_next = scheduler.alphas_cumprod.to(device)[next_t_batch].view(-1, 1, 1, 1)
            x = torch.sqrt(alpha_prod_t_next) * x_0 + torch.sqrt(1 - alpha_prod_t_next) * noise
        else:
            x = x_0

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
        # fake_imgs = generate_samples(student_unet, scheduler, current_batch_size, device, steps=steps)
        fake_imgs = generate_multistep_consistency(student_unet, scheduler, current_batch_size, device, steps=steps)
        # fake_imgs = generate_1_step_samples(student_unet, scheduler, current_batch_size, device)
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
    scheduler.set_timesteps(100) 
    timesteps = scheduler.timesteps.to(device)

    print("Initializing Lightweight Student...")
    student_config = teacher_unet.config.copy()
    student_config["block_out_channels"] = (64, 128, 128, 128) 
    student_unet = UNet2DModel.from_config(student_config).to(device)
    student_unet.train()

    # --- ADDED: Multiple Hooks and Projectors ---
    # We hook the mid_block and the first two up_blocks.
    # Student channels: 128 -> Teacher channels: 256 for these specific blocks
    teacher_hooks = [
        FeatureHook(teacher_unet.mid_block),
        FeatureHook(teacher_unet.up_blocks[1]),
        FeatureHook(teacher_unet.up_blocks[2])
    ]
    student_hooks = [
        FeatureHook(student_unet.mid_block),
        FeatureHook(student_unet.up_blocks[1]),
        FeatureHook(student_unet.up_blocks[2])
    ]
    
    projectors = nn.ModuleList([
        nn.Conv2d(128, 256, kernel_size=1),
        nn.Conv2d(128, 256, kernel_size=1),
        nn.Conv2d(128, 256, kernel_size=1)
    ]).to(device)
    projectors.train()
    # --------------------------------------------

    BATCH_SIZE = 64
    # BATCH_SIZE = 512
    PHASE_1_EPOCHS = 30
    PHASE_2_EPOCHS = 30
    LR = 1e-4

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    
    optimizer = torch.optim.AdamW(list(student_unet.parameters()) + list(projectors.parameters()), lr=LR)

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
            
            # FIX 3: Train on the full continuous timeline, not just the 50 discret steps
            actual_t = torch.randint(0, scheduler.config.num_train_timesteps, (B,), device=device).long()

            alphas_cumprod_device = scheduler.alphas_cumprod.to(device)
            noise = torch.randn_like(imgs)
            alphas_cumprod_t = alphas_cumprod_device[actual_t].view(-1, 1, 1, 1)
            x_t = torch.sqrt(alphas_cumprod_t) * imgs + torch.sqrt(1 - alphas_cumprod_t) * noise

            with torch.no_grad():
                teacher_noise_pred = teacher_unet(x_t, actual_t).sample

            student_noise_pred = student_unet(x_t, actual_t).sample

            # FIX 2: Calculate feature loss using Cosine Similarity to prevent magnitude explosion
            loss_noise = F.mse_loss(student_noise_pred, teacher_noise_pred)
            
            loss_feature = 0.0
            for s_hook, t_hook, proj in zip(student_hooks, teacher_hooks, projectors):
                s_feat = proj(s_hook.features)
                t_feat = t_hook.features
                # Cosine similarity (1 - cos_sim) is robust to large scale differences
                loss_feature += (1.0 - F.cosine_similarity(s_feat.flatten(1), t_feat.flatten(1)).mean())
            loss_feature = loss_feature / len(projectors)
            
            # Since feature loss is now normalized via cosine similarity, a weight of 0.5 or 1.0 is perfectly safe
            loss = loss_noise + (0.5 * loss_feature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix({'Noise': f"{loss_noise.item():.3f}", 'Feat': f"{loss_feature.item():.3f}"})

        if (epoch + 1) % 5 == 0:
            evaluate_student(student_unet, scheduler, dataloader, device, num_samples=2048, batch_size=BATCH_SIZE, steps=100)

    print("\nPhase 1 Complete!")
    torch.save(student_unet.state_dict(), "student_unet_phase1_complete.pt")

    # ==========================================
    # PHASE 2: Consistency Distillation (Learn to be Fast)
    # ==========================================
    print("\n" + "="*40)
    print("PHASE 2: CONSISTENCY DISTILLATION")
    print("="*40)
    
    target_unet = copy.deepcopy(student_unet)
    target_unet.eval()
    for param in target_unet.parameters(): param.requires_grad = False
    
    EMA_RATE = 0.95

    for epoch in range(PHASE_2_EPOCHS):
        pbar = tqdm(dataloader, desc=f"Phase 2 - Epoch {epoch+1}/{PHASE_2_EPOCHS}")
        for imgs, _ in pbar:
            imgs = imgs.to(device)
            B = imgs.shape[0]

            step_indices = torch.randint(0, len(timesteps) - 1, (B,), device=device)
            t_n_plus_1 = timesteps[step_indices]
            t_n = timesteps[step_indices + 1]

            alphas_cumprod_device = scheduler.alphas_cumprod.to(device)
            noise = torch.randn_like(imgs)
            alphas_cumprod_n_plus_1 = alphas_cumprod_device[t_n_plus_1].view(-1, 1, 1, 1)
            x_t_n_plus_1 = torch.sqrt(alphas_cumprod_n_plus_1) * imgs + torch.sqrt(1 - alphas_cumprod_n_plus_1) * noise

            with torch.no_grad():
                teacher_noise_pred = teacher_unet(x_t_n_plus_1, t_n_plus_1).sample
                alpha_prod_t = alphas_cumprod_device[t_n_plus_1].view(-1, 1, 1, 1)
                alpha_prod_t_prev = alphas_cumprod_device[t_n].view(-1, 1, 1, 1)
                
                pred_x0_teacher = (x_t_n_plus_1 - torch.sqrt(1 - alpha_prod_t) * teacher_noise_pred) / torch.sqrt(alpha_prod_t)
                dir_xt_teacher = torch.sqrt(1 - alpha_prod_t_prev) * teacher_noise_pred
                x_t_n = torch.sqrt(alpha_prod_t_prev) * pred_x0_teacher + dir_xt_teacher

            student_noise_pred = student_unet(x_t_n_plus_1, t_n_plus_1).sample
            student_pred_x0 = get_predicted_x0(student_unet, scheduler, x_t_n_plus_1, t_n_plus_1, student_noise_pred)

            with torch.no_grad():
                target_noise_pred = target_unet(x_t_n, t_n).sample
                target_pred_x0 = get_predicted_x0(target_unet, scheduler, x_t_n, t_n, target_noise_pred)

            loss_consistency = F.smooth_l1_loss(student_pred_x0, target_pred_x0)
            
            # Compute multi-level feature loss again
            loss_feature = 0.0
            for s_hook, t_hook, proj in zip(student_hooks, teacher_hooks, projectors):
                s_feat = proj(s_hook.features)
                t_feat = t_hook.features
                loss_feature += (1.0 - F.cosine_similarity(s_feat.flatten(1), t_feat.flatten(1)).mean())
            loss_feature = loss_feature / len(projectors)

            loss = loss_consistency + (0.1 * loss_feature) 

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                for param_student, param_target in zip(student_unet.parameters(), target_unet.parameters()):
                    param_target.data.mul_(EMA_RATE).add_(param_student.data, alpha=1 - EMA_RATE)

            pbar.set_postfix({'Consist': f"{loss_consistency.item():.3f}", 'Feat': f"{loss_feature.item():.3f}"})

        if (epoch + 1) % 5 == 0:
            evaluate_student(student_unet, scheduler, dataloader, device, num_samples=2048, batch_size=BATCH_SIZE, steps=5)

    # Clean up all hooks
    for h in teacher_hooks + student_hooks:
        h.close()
        
    torch.save(student_unet.state_dict(), "student_unet_phase2_final.pt")
    print("Training Complete!")



def test_saved_model(checkpoint_path="student_unet_phase1_complete.pt", batch_size=512):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Setup the Teacher/Scheduler (needed for the config and timesteps)
    teacher_pipe = DDPMPipeline.from_pretrained("google/ddpm-cifar10-32").to(device)
    scheduler = DDIMScheduler.from_config(teacher_pipe.scheduler.config)
    
    # 2. Initialize Student architecture (MUST match the one used during training)
    student_config = teacher_pipe.unet.config.copy()
    student_config["block_out_channels"] = (64, 128, 128, 128) # Matches your training
    student_unet = UNet2DModel.from_config(student_config).to(device)

    # 3. Load the weights
    print(f"Loading weights from {checkpoint_path}...")
    student_unet.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    student_unet.eval()

    # 4. Prepare Dataset for FID (Real images vs Fake images)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,)

    # 5. Run Evaluation
    # Note: 10k samples is the gold standard for FID, but it takes time
    evaluate_student(student_unet, scheduler, dataloader, device, num_samples=10_000, steps=5, batch_size=batch_size)

if __name__ == "__main__":
    # Specify which phase you want to test
    # test_saved_model("student_unet_phase1_complete.pt", batch_size=256)
    # test_saved_model("student_unet_phase2_final.pt", batch_size=256)
    run_two_phase_distillation()

# ========================================
# PHASE 1: STANDARD KNOWLEDGE DISTILLATION
# ========================================
# Phase 1 - Epoch 1/30: 100%|███████████████████████████████████████████████████████████| 781/781 [04:43<00:00,  2.75it/s, Noise=0.013, Feat=0.336]
# Phase 1 - Epoch 2/30: 100%|███████████████████████████████████████████████████████████| 781/781 [04:43<00:00,  2.76it/s, Noise=0.015, Feat=0.240]
# Phase 1 - Epoch 3/30: 100%|███████████████████████████████████████████████████████████| 781/781 [04:42<00:00,  2.76it/s, Noise=0.008, Feat=0.216]
# Phase 1 - Epoch 4/30: 100%|███████████████████████████████████████████████████████████| 781/781 [04:42<00:00,  2.76it/s, Noise=0.008, Feat=0.194]
# Phase 1 - Epoch 5/30: 100%|███████████████████████████████████████████████████████████| 781/781 [04:42<00:00,  2.76it/s, Noise=0.005, Feat=0.183]

# --- Evaluating on 2048 Samples (50 Steps) ---
# C:\Users\Dimitar Trajkov\anaconda3\Lib\site-packages\torchmetrics\utilities\prints.py:43: UserWarning: Metric `InceptionScore` will save all extracted features in buffer. For large datasets this may lead to large memory footprint.
#   warnings.warn(*args, **kwargs)
# Generating 50-Step Samples: 100%|████████████████████████████████████████████████████████████████████████████| 2048/2048 [01:36<00:00, 21.27it/s]Results -> FID: 129.3135 | IS: 4.2815 ± 0.1851
# Generating 50-Step Samples: 100%|████████████████████████████████████████████████████████████████████████████| 2048/2048 [01:40<00:00, 20.35it/s]
# Phase 1 - Epoch 6/30: 100%|███████████████████████████████████████████████████████████| 781/781 [04:41<00:00,  2.77it/s, Noise=0.007, Feat=0.207]
# Phase 1 - Epoch 7/30: 100%|███████████████████████████████████████████████████████████| 781/781 [04:43<00:00,  2.76it/s, Noise=0.004, Feat=0.230]
# Phase 1 - Epoch 8/30: 100%|███████████████████████████████████████████████████████████| 781/781 [04:44<00:00,  2.74it/s, Noise=0.006, Feat=0.171]
# Phase 1 - Epoch 9/30: 100%|███████████████████████████████████████████████████████████| 781/781 [04:43<00:00,  2.76it/s, Noise=0.003, Feat=0.178]
# Phase 1 - Epoch 10/30: 100%|██████████████████████████████████████████████████████████| 781/781 [04:40<00:00,  2.78it/s, Noise=0.006, Feat=0.188]

# --- Evaluating on 2048 Samples (50 Steps) ---
# Generating 50-Step Samples: 100%|████████████████████████████████████████████████████████████████████████████| 2048/2048 [01:38<00:00, 21.13it/s]Results -> FID: 104.1442 | IS: 5.5335 ± 0.2176
# Generating 50-Step Samples: 100%|████████████████████████████████████████████████████████████████████████████| 2048/2048 [01:42<00:00, 20.05it/s]
# Phase 1 - Epoch 11/30: 100%|██████████████████████████████████████████████████████████| 781/781 [04:41<00:00,  2.77it/s, Noise=0.006, Feat=0.208]
# Phase 1 - Epoch 12/30: 100%|██████████████████████████████████████████████████████████| 781/781 [04:43<00:00,  2.75it/s, Noise=0.008, Feat=0.191]
# Phase 1 - Epoch 13/30: 100%|██████████████████████████████████████████████████████████| 781/781 [04:44<00:00,  2.74it/s, Noise=0.003, Feat=0.183]
# Phase 1 - Epoch 14/30: 100%|██████████████████████████████████████████████████████████| 781/781 [04:43<00:00,  2.75it/s, Noise=0.004, Feat=0.164]
# Phase 1 - Epoch 15/30: 100%|██████████████████████████████████████████████████████████| 781/781 [04:43<00:00,  2.75it/s, Noise=0.003, Feat=0.178]

# --- Evaluating on 2048 Samples (50 Steps) ---
# Generating 50-Step Samples: 100%|████████████████████████████████████████████████████████████████████████████| 2048/2048 [01:42<00:00, 20.08it/s]Results -> FID: 85.8316 | IS: 5.1974 ± 0.2395
# Generating 50-Step Samples: 100%|████████████████████████████████████████████████████████████████████████████| 2048/2048 [01:45<00:00, 19.37it/s]
# Phase 1 - Epoch 16/30: 100%|██████████████████████████████████████████████████████████| 781/781 [04:42<00:00,  2.77it/s, Noise=0.007, Feat=0.181]
# Phase 1 - Epoch 17/30: 100%|██████████████████████████████████████████████████████████| 781/781 [04:43<00:00,  2.75it/s, Noise=0.004, Feat=0.165]
# Phase 1 - Epoch 18/30: 100%|██████████████████████████████████████████████████████████| 781/781 [04:43<00:00,  2.75it/s, Noise=0.004, Feat=0.186]
# Phase 1 - Epoch 19/30: 100%|██████████████████████████████████████████████████████████| 781/781 [04:39<00:00,  2.80it/s, Noise=0.003, Feat=0.157]
# Phase 1 - Epoch 20/30: 100%|██████████████████████████████████████████████████████████| 781/781 [04:38<00:00,  2.80it/s, Noise=0.002, Feat=0.150]

# --- Evaluating on 2048 Samples (50 Steps) ---
# Generating 50-Step Samples: 100%|████████████████████████████████████████████████████████████████████████████| 2048/2048 [01:35<00:00, 21.60it/s]Results -> FID: 165.8118 | IS: 3.3073 ± 0.2293
# Generating 50-Step Samples: 100%|████████████████████████████████████████████████████████████████████████████| 2048/2048 [01:38<00:00, 20.85it/s]
# Phase 1 - Epoch 21/30: 100%|██████████████████████████████████████████████████████████| 781/781 [04:38<00:00,  2.80it/s, Noise=0.003, Feat=0.163]
# Phase 1 - Epoch 22/30: 100%|██████████████████████████████████████████████████████████| 781/781 [04:40<00:00,  2.78it/s, Noise=0.006, Feat=0.206]
# Phase 1 - Epoch 23/30: 100%|██████████████████████████████████████████████████████████| 781/781 [04:41<00:00,  2.77it/s, Noise=0.007, Feat=0.176]
# Phase 1 - Epoch 24/30: 100%|██████████████████████████████████████████████████████████| 781/781 [04:43<00:00,  2.75it/s, Noise=0.004, Feat=0.143]
# Phase 1 - Epoch 25/30: 100%|██████████████████████████████████████████████████████████| 781/781 [04:43<00:00,  2.75it/s, Noise=0.003, Feat=0.137]

# --- Evaluating on 2048 Samples (50 Steps) ---
# Generating 50-Step Samples: 100%|████████████████████████████████████████████████████████████████████████████| 2048/2048 [01:39<00:00, 20.47it/s]Results -> FID: 76.7583 | IS: 5.6832 ± 0.3478
# Generating 50-Step Samples: 100%|████████████████████████████████████████████████████████████████████████████| 2048/2048 [01:42<00:00, 19.98it/s]
# Phase 1 - Epoch 26/30: 100%|██████████████████████████████████████████████████████████| 781/781 [04:44<00:00,  2.75it/s, Noise=0.003, Feat=0.161]
# Phase 1 - Epoch 27/30: 100%|██████████████████████████████████████████████████████████| 781/781 [04:43<00:00,  2.75it/s, Noise=0.003, Feat=0.157]
# Phase 1 - Epoch 28/30: 100%|██████████████████████████████████████████████████████████| 781/781 [04:43<00:00,  2.75it/s, Noise=0.002, Feat=0.145]
# Phase 1 - Epoch 29/30: 100%|██████████████████████████████████████████████████████████| 781/781 [04:43<00:00,  2.75it/s, Noise=0.003, Feat=0.164]
# Phase 1 - Epoch 30/30: 100%|██████████████████████████████████████████████████████████| 781/781 [04:43<00:00,  2.75it/s, Noise=0.004, Feat=0.147]

# --- Evaluating on 2048 Samples (50 Steps) ---
# Generating 50-Step Samples: 100%|████████████████████████████████████████████████████████████████████████████| 2048/2048 [01:43<00:00, 20.24it/s]Results -> FID: 67.0487 | IS: 5.6006 ± 0.2423
# Generating 50-Step Samples: 100%|████████████████████████████████████████████████████████████████████████████| 2048/2048 [01:46<00:00, 19.24it/s]

# Phase 1 Complete!

# ========================================
# PHASE 2: CONSISTENCY DISTILLATION
# ========================================
# Phase 2 - Epoch 1/30: 100%|█████████████████████████████████████████████████████████| 781/781 [05:37<00:00,  2.31it/s, Consist=0.019, Feat=0.163]
# Phase 2 - Epoch 2/30: 100%|█████████████████████████████████████████████████████████| 781/781 [05:37<00:00,  2.31it/s, Consist=0.061, Feat=0.141]
# Phase 2 - Epoch 3/30: 100%|█████████████████████████████████████████████████████████| 781/781 [05:36<00:00,  2.32it/s, Consist=0.016, Feat=0.132]
# Phase 2 - Epoch 4/30: 100%|█████████████████████████████████████████████████████████| 781/781 [05:35<00:00,  2.33it/s, Consist=0.003, Feat=0.168]
# Phase 2 - Epoch 5/30: 100%|█████████████████████████████████████████████████████████| 781/781 [05:34<00:00,  2.33it/s, Consist=0.058, Feat=0.156]

# --- Evaluating on 2048 Samples (100 Steps) ---
# Generating 100-Step Samples: 100%|███████████████████████████████████████████████████████████████████████████| 2048/2048 [03:07<00:00, 11.12it/s]Results -> FID: 261.7758 | IS: 2.0477 ± 0.0671
# Generating 100-Step Samples: 100%|███████████████████████████████████████████████████████████████████████████| 2048/2048 [03:11<00:00, 10.69it/s]
# Phase 2 - Epoch 6/30: 100%|█████████████████████████████████████████████████████████| 781/781 [05:38<00:00,  2.31it/s, Consist=0.012, Feat=0.147]
# Phase 2 - Epoch 7/30: 100%|█████████████████████████████████████████████████████████| 781/781 [05:36<00:00,  2.32it/s, Consist=0.113, Feat=0.153]
# Phase 2 - Epoch 8/30: 100%|█████████████████████████████████████████████████████████| 781/781 [05:33<00:00,  2.35it/s, Consist=0.008, Feat=0.158]
# Phase 2 - Epoch 9/30: 100%|█████████████████████████████████████████████████████████| 781/781 [05:33<00:00,  2.34it/s, Consist=0.027, Feat=0.139]
# Phase 2 - Epoch 10/30: 100%|████████████████████████████████████████████████████████| 781/781 [53:39<00:00,  4.12s/it, Consist=0.002, Feat=0.183]

# --- Evaluating on 2048 Samples (100 Steps) ---
# Generating 100-Step Samples: 100%|███████████████████████████████████████████████████████████████████████████| 2048/2048 [03:12<00:00, 10.68it/s]Results -> FID: 282.8624 | IS: 1.8236 ± 0.0404
# Generating 100-Step Samples: 100%|███████████████████████████████████████████████████████████████████████████| 2048/2048 [03:15<00:00, 10.48it/s]
# Phase 2 - Epoch 11/30: 100%|████████████████████████████████████████████████████████| 781/781 [05:38<00:00,  2.30it/s, Consist=0.002, Feat=0.165]
# Phase 2 - Epoch 12/30: 100%|████████████████████████████████████████████████████████| 781/781 [05:39<00:00,  2.30it/s, Consist=0.002, Feat=0.183]
# Phase 2 - Epoch 13/30: 100%|████████████████████████████████████████████████████████| 781/781 [05:37<00:00,  2.31it/s, Consist=0.005, Feat=0.153]
# Phase 2 - Epoch 14/30: 100%|████████████████████████████████████████████████████████| 781/781 [05:37<00:00,  2.31it/s, Consist=0.004, Feat=0.170]
# Phase 2 - Epoch 15/30: 100%|████████████████████████████████████████████████████████| 781/781 [05:35<00:00,  2.33it/s, Consist=0.033, Feat=0.186]

# --- Evaluating on 2048 Samples (100 Steps) ---
# Generating 100-Step Samples: 100%|███████████████████████████████████████████████████████████████████████████| 2048/2048 [03:06<00:00, 10.99it/s]Results -> FID: 332.7444 | IS: 1.4383 ± 0.0281
# Generating 100-Step Samples: 100%|███████████████████████████████████████████████████████████████████████████| 2048/2048 [03:10<00:00, 10.74it/s]
# Phase 2 - Epoch 16/30: 100%|████████████████████████████████████████████████████████| 781/781 [05:34<00:00,  2.34it/s, Consist=0.003, Feat=0.149]
# Phase 2 - Epoch 17/30: 100%|████████████████████████████████████████████████████████| 781/781 [05:34<00:00,  2.33it/s, Consist=0.003, Feat=0.123]
# Phase 2 - Epoch 18/30: 100%|████████████████████████████████████████████████████████| 781/781 [05:34<00:00,  2.33it/s, Consist=0.003, Feat=0.137]
# Phase 2 - Epoch 19/30: 100%|████████████████████████████████████████████████████████| 781/781 [05:34<00:00,  2.34it/s, Consist=0.004, Feat=0.139]
# Phase 2 - Epoch 20/30: 100%|████████████████████████████████████████████████████████| 781/781 [05:31<00:00,  2.35it/s, Consist=0.001, Feat=0.165]

# --- Evaluating on 2048 Samples (100 Steps) ---
# Generating 100-Step Samples: 100%|███████████████████████████████████████████████████████████████████████████| 2048/2048 [04:16<00:00,  7.74it/s]Results -> FID: 343.4049 | IS: 1.4770 ± 0.0267
# Generating 100-Step Samples: 100%|███████████████████████████████████████████████████████████████████████████| 2048/2048 [04:20<00:00,  7.87it/s]
# Phase 2 - Epoch 21/30: 100%|████████████████████████████████████████████████████████| 781/781 [06:52<00:00,  1.89it/s, Consist=0.017, Feat=0.171]
# Phase 2 - Epoch 22/30: 100%|████████████████████████████████████████████████████████| 781/781 [07:06<00:00,  1.83it/s, Consist=0.002, Feat=0.160]
# Phase 2 - Epoch 23/30: 100%|████████████████████████████████████████████████████████| 781/781 [07:07<00:00,  1.83it/s, Consist=0.002, Feat=0.162]
# Phase 2 - Epoch 24/30: 100%|████████████████████████████████████████████████████████| 781/781 [07:06<00:00,  1.83it/s, Consist=0.010, Feat=0.150]
# Phase 2 - Epoch 25/30: 100%|████████████████████████████████████████████████████████| 781/781 [07:05<00:00,  1.83it/s, Consist=0.011, Feat=0.153]

# --- Evaluating on 2048 Samples (100 Steps) ---
# Generating 100-Step Samples: 100%|███████████████████████████████████████████████████████████████████████████| 2048/2048 [06:34<00:00,  5.52it/s]Results -> FID: 328.0770 | IS: 1.5661 ± 0.0385
# Generating 100-Step Samples: 100%|███████████████████████████████████████████████████████████████████████████| 2048/2048 [06:38<00:00,  5.14it/s]
# Phase 2 - Epoch 26/30: 100%|████████████████████████████████████████████████████████| 781/781 [07:57<00:00,  1.64it/s, Consist=0.005, Feat=0.145]
# Phase 2 - Epoch 27/30: 100%|████████████████████████████████████████████████████████| 781/781 [07:52<00:00,  1.65it/s, Consist=0.003, Feat=0.155]
# Phase 2 - Epoch 28/30: 100%|████████████████████████████████████████████████████████| 781/781 [07:55<00:00,  1.64it/s, Consist=0.005, Feat=0.134]
# Phase 2 - Epoch 29/30: 100%|████████████████████████████████████████████████████████| 781/781 [07:52<00:00,  1.65it/s, Consist=0.004, Feat=0.156]
# Phase 2 - Epoch 30/30: 100%|████████████████████████████████████████████████████████| 781/781 [08:04<00:00,  1.61it/s, Consist=0.003, Feat=0.176]

# --- Evaluating on 2048 Samples (100 Steps) ---
# Generating 100-Step Samples: 100%|███████████████████████████████████████████████████████████████████████████| 2048/2048 [06:48<00:00,  4.95it/s]Results -> FID: 339.3658 | IS: 1.5160 ± 0.0207
# Generating 100-Step Samples: 100%|███████████████████████████████████████████████████████████████████████████| 2048/2048 [06:52<00:00,  4.96it/s]
# Training Complete!



# PHASE 1 30 epochs
# Generating 100-Step Samples: 100%|█████████████████████████████████████████████████████████████████████████| 10000/10000 [15:28<00:00,  9.14it/s]Results -> FID: 48.8848 | IS: 6.0286 ± 0.1779
# Generating 20-Step Samples: 100%|██████████████████████████████████████████████████████████████████████████| 10000/10000 [04:48<00:00, 28.95it/s]Results -> FID: 50.5275 | IS: 5.8279 ± 0.0770
# Generating 5-Step Samples: 100%|███████████████████████████████████████████████████████████████████████████| 10000/10000 [02:34<00:00, 50.36it/s]Results -> FID: 158.9064 | IS: 3.2868 ± 0.0718
# Generating 1-Step Samples: 100%|███████████████████████████████████████████████████████████████████████████| 10000/10000 [01:57<00:00, 62.74it/s]Results -> FID: 383.2865 | IS: 1.3950 ± 0.0077

# PHASE 2 30 epochs
# Generating 1-Step Samples: 100%|███████████████████████████████████████████████████████████████████████████| 10000/10000 [02:01<00:00, 59.84it/s]Results -> FID: 357.5378 | IS: 1.5389 ± 0.0111
# Generating 5-Step Samples: 100%|███████████████████████████████████████████████████████████████████████████| 10000/10000 [02:36<00:00, 49.03it/s]Results -> FID: 288.8138 | IS: 1.6810 ± 0.0235