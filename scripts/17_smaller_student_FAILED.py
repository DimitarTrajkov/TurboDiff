import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from diffusers import DDPMPipeline, DDIMScheduler, UNet2DModel
import copy
import time
from tqdm import tqdm
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore

# ==========================================
# 1. Feature Extraction Hooks & Projector
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
    pred_x0 = (x_t - torch.sqrt(beta_prod_t) * noise_pred) / torch.sqrt(alpha_prod_t)
    return pred_x0

# ==========================================
# 2. 5-Step Evaluation Functions
# ==========================================
@torch.no_grad()
def generate_5_step_samples(unet, scheduler, batch_size, device):
    """Generates a batch using 5-Step Consistency Sampling."""
    steps = 5
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
def evaluate_student(student_unet, scheduler, dataloader, device, num_samples=2048, batch_size=64):
    """Calculates FID and IS using 5-step generation."""
    print(f"\n--- Running 5-Step Evaluation on {num_samples} Samples ---")
    student_unet.eval()
    
    fid_metric = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    is_metric = InceptionScore(normalize=True).to(device)
    
    # 1. Collect Real Images for FID
    real_processed = 0
    for imgs, _ in dataloader:
        if real_processed >= num_samples:
            break
        # Process what we need to hit exactly num_samples
        batch_imgs = imgs[:min(batch_size, num_samples - real_processed)].to(device)
        # Convert [-1, 1] to [0, 1] for torchmetrics normalize=True
        batch_imgs = (batch_imgs + 1.0) / 2.0 
        fid_metric.update(batch_imgs, real=True)
        real_processed += batch_imgs.shape[0]

    # 2. Generate Fake Images for FID and IS
    fake_processed = 0
    pbar = tqdm(total=num_samples, desc="Generating 5-Step Samples")
    while fake_processed < num_samples:
        current_batch_size = min(batch_size, num_samples - fake_processed)
        fake_imgs = generate_5_step_samples(student_unet, scheduler, current_batch_size, device)
        
        # Convert [-1, 1] to [0, 1]
        fake_imgs = (fake_imgs + 1.0) / 2.0 
        
        fid_metric.update(fake_imgs, real=False)
        is_metric.update(fake_imgs)
        
        fake_processed += current_batch_size
        pbar.update(current_batch_size)
    
    # 3. Compute Metrics
    print("Computing metrics (this may take a moment)...")
    fid_score = fid_metric.compute().item()
    is_score_mean, is_score_std = is_metric.compute()
    
    print(f"Results -> FID: {fid_score:.4f} | IS: {is_score_mean.item():.4f} ± {is_score_std.item():.4f}")
    
    # Reset metrics for the next evaluation
    fid_metric.reset()
    is_metric.reset()
    student_unet.train()
    
    return fid_score

# ==========================================
# 3. Main Training Loop
# ==========================================
def train_consistency_feature_distillation():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    print("Loading Teacher Model...")
    teacher_pipe = DDPMPipeline.from_pretrained("google/ddpm-cifar10-32").to(device)
    teacher_unet = teacher_pipe.unet
    teacher_unet.eval() 
    
    scheduler = DDIMScheduler.from_config(teacher_pipe.scheduler.config)
    scheduler.set_timesteps(50) 
    timesteps = scheduler.timesteps.to(device)

    print("Initializing Lightweight Student...")
    student_config = teacher_unet.config.copy()
    student_config["block_out_channels"] = (64, 128, 128, 128) 
    
    student_unet = UNet2DModel.from_config(student_config).to(device)
    student_unet.train()
    
    target_unet = copy.deepcopy(student_unet)
    target_unet.eval()

    for param in teacher_unet.parameters(): param.requires_grad = False
    for param in target_unet.parameters(): param.requires_grad = False

    projector = FeatureProjector(student_channels=128, teacher_channels=256).to(device)
    projector.train()

    teacher_hook = FeatureHook(teacher_unet.mid_block)
    student_hook = FeatureHook(student_unet.mid_block)

    BATCH_SIZE = 64
    EPOCHS = 50
    LR = 1e-4
    EMA_RATE = 0.95
    FEATURE_LOSS_WEIGHT = 0.5 

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    
    optimizer = torch.optim.AdamW(list(student_unet.parameters()) + list(projector.parameters()), lr=LR)

    print("\nStarting Feature Distillation...")
    for epoch in range(EPOCHS):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
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
            
            projected_student_features = projector(student_hook.features)
            loss_feature = F.mse_loss(projected_student_features, teacher_hook.features)

            loss = loss_consistency + (FEATURE_LOSS_WEIGHT * loss_feature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                for param_student, param_target in zip(student_unet.parameters(), target_unet.parameters()):
                    param_target.data.mul_(EMA_RATE).add_(param_student.data, alpha=1 - EMA_RATE)

            pbar.set_postfix({'Total': f"{loss.item():.3f}", 'Consist': f"{loss_consistency.item():.3f}"})

        # Save Checkpoint
        torch.save(student_unet.state_dict(), f"student_unet_tiny_epoch_{epoch+1}.pt")
        
        # ==========================================
        # Evaluation Trigger: Every 5 Epochs
        # ==========================================
        if (epoch + 1) % 5 == 0:
            evaluate_student(
                student_unet=student_unet, 
                scheduler=scheduler, 
                dataloader=dataloader, 
                device=device, 
                num_samples=2048, 
                batch_size=BATCH_SIZE
            )

    teacher_hook.close()
    student_hook.close()

if __name__ == "__main__":
    train_consistency_feature_distillation()
    
    
    
# (base) PS C:\Users\Dimitar Trajkov\Desktop\SAPIENZA\CV> python .\17_smaller_student.py
# WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
# I0000 00:00:1778067874.409911   28504 port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
# WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
# I0000 00:00:1778067878.660410   28504 port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
# Using device: cuda

# Loading Teacher Model...
# Loading pipeline components...:   0%|                                                                          | 0/2 [00:00<?, ?it/s]An error occurred while trying to fetch C:\Users\Dimitar Trajkov\.cache\huggingface\hub\models--google--ddpm-cifar10-32\snapshots\267b167dc01f0e4e61923ea244e8b988f84deb80: Error no file named diffusion_pytorch_model.safetensors found in directory C:\Users\Dimitar Trajkov\.cache\huggingface\hub\models--google--ddpm-cifar10-32\snapshots\267b167dc01f0e4e61923ea244e8b988f84deb80.
# Defaulting to unsafe serialization. Pass `allow_pickle=False` to raise an error instead.
# Loading pipeline components...: 100%|██████████████████████████████████████████████████████████████████| 2/2 [00:00<00:00,  6.19it/s]
# Initializing Lightweight Student...
# Files already downloaded and verified

# Starting Feature Distillation...
# Epoch 1/50: 100%|█████████████████████████████████████████████████████| 781/781 [05:22<00:00,  2.42it/s, Total=54.436, Consist=0.064]
# Epoch 2/50: 100%|█████████████████████████████████████████████████████| 781/781 [05:19<00:00,  2.44it/s, Total=43.070, Consist=0.040]
# Epoch 3/50: 100%|█████████████████████████████████████████████████████| 781/781 [05:19<00:00,  2.45it/s, Total=42.739, Consist=0.040]
# Epoch 4/50: 100%|█████████████████████████████████████████████████████| 781/781 [05:20<00:00,  2.44it/s, Total=39.960, Consist=0.021]
# Epoch 5/50: 100%|█████████████████████████████████████████████████████| 781/781 [05:15<00:00,  2.47it/s, Total=36.813, Consist=0.028]

# --- Running 5-Step Evaluation on 2048 Samples ---
# C:\Users\Dimitar Trajkov\anaconda3\Lib\site-packages\torchmetrics\utilities\prints.py:43: UserWarning: Metric `InceptionScore` will save all extracted features in buffer. For large datasets this may lead to large memory footprint.
#   warnings.warn(*args, **kwargs)
# Generating 5-Step Samples: 100%|████████████████████████████████████████████████████████████████| 2048/2048 [00:19<00:00, 106.31it/s]Computing metrics (this may take a moment)...
# Results -> FID: 358.3960 | IS: 1.3456 ± 0.0290
# Generating 5-Step Samples: 100%|█████████████████████████████████████████████████████████████████| 2048/2048 [00:23<00:00, 85.38it/s]
# Epoch 6/50: 100%|█████████████████████████████████████████████████████| 781/781 [05:18<00:00,  2.45it/s, Total=33.298, Consist=0.035]
# Epoch 7/50: 100%|█████████████████████████████████████████████████████| 781/781 [05:20<00:00,  2.44it/s, Total=33.956, Consist=0.028]
# Epoch 8/50: 100%|█████████████████████████████████████████████████████| 781/781 [05:20<00:00,  2.44it/s, Total=34.862, Consist=0.049]
# Epoch 9/50: 100%|█████████████████████████████████████████████████████| 781/781 [05:15<00:00,  2.48it/s, Total=33.673, Consist=0.023]
# Epoch 10/50: 100%|████████████████████████████████████████████████████| 781/781 [05:14<00:00,  2.49it/s, Total=38.137, Consist=0.006]

# --- Running 5-Step Evaluation on 2048 Samples ---
# Generating 5-Step Samples: 100%|████████████████████████████████████████████████████████████████| 2048/2048 [00:19<00:00, 104.38it/s]Computing metrics (this may take a moment)...
# Results -> FID: 293.3865 | IS: 1.4413 ± 0.0292
# Generating 5-Step Samples: 100%|█████████████████████████████████████████████████████████████████| 2048/2048 [00:22<00:00, 90.34it/s]
# Epoch 11/50: 100%|████████████████████████████████████████████████████| 781/781 [05:18<00:00,  2.46it/s, Total=27.891, Consist=0.017]
# Epoch 12/50: 100%|████████████████████████████████████████████████████| 781/781 [05:20<00:00,  2.44it/s, Total=30.329, Consist=0.031]
# Epoch 13/50: 100%|████████████████████████████████████████████████████| 781/781 [05:20<00:00,  2.44it/s, Total=31.258, Consist=0.017]
# Epoch 14/50: 100%|████████████████████████████████████████████████████| 781/781 [05:20<00:00,  2.44it/s, Total=26.968, Consist=0.005]
# Epoch 15/50: 100%|████████████████████████████████████████████████████| 781/781 [05:20<00:00,  2.44it/s, Total=31.349, Consist=0.025]

# --- Running 5-Step Evaluation on 2048 Samples ---
# Generating 5-Step Samples: 100%|████████████████████████████████████████████████████████████████| 2048/2048 [00:19<00:00, 104.55it/s]Computing metrics (this may take a moment)...
# Results -> FID: 282.3672 | IS: 1.6076 ± 0.0343
# Generating 5-Step Samples: 100%|█████████████████████████████████████████████████████████████████| 2048/2048 [00:22<00:00, 90.57it/s]
# Epoch 16/50: 100%|████████████████████████████████████████████████████| 781/781 [05:19<00:00,  2.44it/s, Total=32.794, Consist=0.032]
# Epoch 17/50: 100%|████████████████████████████████████████████████████| 781/781 [05:20<00:00,  2.44it/s, Total=30.206, Consist=0.032]
# Epoch 18/50: 100%|████████████████████████████████████████████████████| 781/781 [05:21<00:00,  2.43it/s, Total=33.725, Consist=0.023]
# Epoch 19/50: 100%|████████████████████████████████████████████████████| 781/781 [45:39<00:00,  3.51s/it, Total=27.304, Consist=0.025]
# Epoch 20/50: 100%|████████████████████████████████████████████████████| 781/781 [05:20<00:00,  2.44it/s, Total=31.374, Consist=0.052]

# --- Running 5-Step Evaluation on 2048 Samples ---
# Generating 5-Step Samples: 100%|████████████████████████████████████████████████████████████████| 2048/2048 [00:19<00:00, 101.28it/s]Computing metrics (this may take a moment)...
# Results -> FID: 293.3076 | IS: 1.4687 ± 0.0395
# Generating 5-Step Samples: 100%|█████████████████████████████████████████████████████████████████| 2048/2048 [00:23<00:00, 88.22it/s]
# Epoch 21/50: 100%|████████████████████████████████████████████████████| 781/781 [05:19<00:00,  2.45it/s, Total=29.262, Consist=0.026]
# Epoch 22/50: 100%|████████████████████████████████████████████████████| 781/781 [05:18<00:00,  2.45it/s, Total=31.748, Consist=0.009]
# Epoch 23/50: 100%|████████████████████████████████████████████████████| 781/781 [05:18<00:00,  2.45it/s, Total=30.738, Consist=0.020]
# Epoch 24/50: 100%|████████████████████████████████████████████████████| 781/781 [05:19<00:00,  2.44it/s, Total=28.991, Consist=0.012]
# Epoch 25/50: 100%|████████████████████████████████████████████████████| 781/781 [05:19<00:00,  2.44it/s, Total=30.441, Consist=0.017]

# --- Running 5-Step Evaluation on 2048 Samples ---
# Generating 5-Step Samples: 100%|████████████████████████████████████████████████████████████████| 2048/2048 [00:19<00:00, 102.43it/s]Computing metrics (this may take a moment)...
# Results -> FID: 290.0076 | IS: 1.5504 ± 0.0364
# Generating 5-Step Samples: 100%|█████████████████████████████████████████████████████████████████| 2048/2048 [00:23<00:00, 87.05it/s]
# Epoch 26/50: 100%|████████████████████████████████████████████████████| 781/781 [05:20<00:00,  2.44it/s, Total=27.915, Consist=0.014]
# Epoch 27/50: 100%|████████████████████████████████████████████████████| 781/781 [05:21<00:00,  2.43it/s, Total=30.118, Consist=0.046]
# Epoch 28/50: 100%|████████████████████████████████████████████████████| 781/781 [05:20<00:00,  2.43it/s, Total=29.577, Consist=0.012]
# Epoch 29/50: 100%|████████████████████████████████████████████████████| 781/781 [05:23<00:00,  2.41it/s, Total=28.002, Consist=0.012]
# Epoch 30/50: 100%|████████████████████████████████████████████████████| 781/781 [05:21<00:00,  2.43it/s, Total=29.226, Consist=0.012]

# --- Running 5-Step Evaluation on 2048 Samples ---
# Generating 5-Step Samples: 100%|█████████████████████████████████████████████████████████████████| 2048/2048 [00:20<00:00, 96.51it/s]Computing metrics (this may take a moment)...
# Results -> FID: 288.4526 | IS: 1.6825 ± 0.0356
# Generating 5-Step Samples: 100%|█████████████████████████████████████████████████████████████████| 2048/2048 [00:24<00:00, 84.50it/s]
# Epoch 31/50: 100%|████████████████████████████████████████████████████| 781/781 [05:20<00:00,  2.44it/s, Total=33.522, Consist=0.008]
# Epoch 32/50: 100%|████████████████████████████████████████████████████| 781/781 [05:20<00:00,  2.44it/s, Total=28.875, Consist=0.003]
# Epoch 33/50: 100%|████████████████████████████████████████████████████| 781/781 [05:20<00:00,  2.44it/s, Total=27.305, Consist=0.013]
# Epoch 34/50: 100%|████████████████████████████████████████████████████| 781/781 [05:18<00:00,  2.45it/s, Total=33.159, Consist=0.008]
# Epoch 35/50: 100%|████████████████████████████████████████████████████| 781/781 [05:20<00:00,  2.44it/s, Total=30.211, Consist=0.005]

# --- Running 5-Step Evaluation on 2048 Samples ---
# Generating 5-Step Samples: 100%|████████████████████████████████████████████████████████████████| 2048/2048 [00:19<00:00, 101.85it/s]Computing metrics (this may take a moment)...
# Results -> FID: 295.5233 | IS: 1.4859 ± 0.0295
# Generating 5-Step Samples: 100%|█████████████████████████████████████████████████████████████████| 2048/2048 [00:23<00:00, 88.96it/s]
# Epoch 36/50: 100%|████████████████████████████████████████████████████| 781/781 [34:34<00:00,  2.66s/it, Total=31.733, Consist=0.009]
# Epoch 37/50: 100%|████████████████████████████████████████████████████| 781/781 [05:15<00:00,  2.47it/s, Total=31.194, Consist=0.005]
# Epoch 38/50: 100%|████████████████████████████████████████████████████| 781/781 [05:12<00:00,  2.50it/s, Total=26.360, Consist=0.013]
# Epoch 39/50: 100%|████████████████████████████████████████████████████| 781/781 [05:14<00:00,  2.48it/s, Total=28.674, Consist=0.023]
# Epoch 40/50: 100%|████████████████████████████████████████████████████| 781/781 [05:12<00:00,  2.50it/s, Total=30.051, Consist=0.007]

# --- Running 5-Step Evaluation on 2048 Samples ---
# Generating 5-Step Samples: 100%|████████████████████████████████████████████████████████████████| 2048/2048 [00:19<00:00, 106.86it/s]Computing metrics (this may take a moment)...
# Results -> FID: 293.3966 | IS: 1.6022 ± 0.0305
# Generating 5-Step Samples: 100%|█████████████████████████████████████████████████████████████████| 2048/2048 [00:22<00:00, 90.60it/s]
# Epoch 41/50: 100%|████████████████████████████████████████████████████| 781/781 [05:12<00:00,  2.50it/s, Total=29.425, Consist=0.007]
# Epoch 42/50: 100%|████████████████████████████████████████████████████| 781/781 [05:11<00:00,  2.50it/s, Total=26.662, Consist=0.006]
# Epoch 43/50: 100%|████████████████████████████████████████████████████| 781/781 [05:12<00:00,  2.50it/s, Total=25.283, Consist=0.005]
# Epoch 44/50: 100%|████████████████████████████████████████████████████| 781/781 [05:11<00:00,  2.50it/s, Total=29.869, Consist=0.008]
# Epoch 45/50: 100%|████████████████████████████████████████████████████| 781/781 [05:12<00:00,  2.50it/s, Total=25.888, Consist=0.008]

# --- Running 5-Step Evaluation on 2048 Samples ---
# Generating 5-Step Samples: 100%|████████████████████████████████████████████████████████████████| 2048/2048 [00:19<00:00, 106.23it/s]Computing metrics (this may take a moment)...
# Results -> FID: 294.0448 | IS: 1.6589 ± 0.0437
# Generating 5-Step Samples: 100%|█████████████████████████████████████████████████████████████████| 2048/2048 [00:22<00:00, 91.64it/s]
# Epoch 46/50:  96%|█████████████████████████████████████████████████▊  | 749/781 [04:59<00:13,  2.38it/s, Total=28.604, Consist=0.010]