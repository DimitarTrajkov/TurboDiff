import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from diffusers import DDPMPipeline, DDIMScheduler
import copy
from tqdm import tqdm

def get_predicted_x0(unet, scheduler, x_t, t, noise_pred):
    """
    Standard diffusers UNets predict the noise (epsilon).
    This function uses the scheduler's math to convert that predicted noise
    into a prediction of the final clean image (x0).
    """
    # FIX: Send the scheduler's tensor to the GPU before indexing
    alphas_cumprod_device = scheduler.alphas_cumprod.to(x_t.device)
    
    alpha_prod_t = alphas_cumprod_device[t]
    beta_prod_t = 1 - alpha_prod_t
    
    # Reshape for broadcasting
    alpha_prod_t = alpha_prod_t.view(-1, 1, 1, 1)
    beta_prod_t = beta_prod_t.view(-1, 1, 1, 1)
    
    # Mathematical inversion: x0 = (x_t - sqrt(1 - alpha) * noise) / sqrt(alpha)
    pred_x0 = (x_t - torch.sqrt(beta_prod_t) * noise_pred) / torch.sqrt(alpha_prod_t)
    return pred_x0

def train_consistency_distillation():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    # ==========================================
    # 1. Load the Teacher Model
    # ==========================================
    print("Loading Teacher Model...")
    teacher_pipe = DDPMPipeline.from_pretrained("google/ddpm-cifar10-32").to(device)
    teacher_unet = teacher_pipe.unet
    teacher_unet.eval() # Teacher is frozen
    
    # We use DDIM for the teacher steps to allow skipping
    scheduler = DDIMScheduler.from_config(teacher_pipe.scheduler.config)
    scheduler.set_timesteps(50) # The teacher will operate on a 50-step trajectory
    timesteps = scheduler.timesteps.to(device)

    # ==========================================
    # 2. Initialize Student & Target Models
    # ==========================================
    print("Initializing Student and Target Networks...")
    # Student starts as a perfect clone of the Teacher
    student_unet = copy.deepcopy(teacher_unet)
    student_unet.train()
    
    # Target starts as a perfect clone of the Student
    target_unet = copy.deepcopy(student_unet)
    target_unet.eval() # Target is frozen and updated via EMA

    # Turn off gradients for Teacher and Target
    for param in teacher_unet.parameters(): param.requires_grad = False
    for param in target_unet.parameters(): param.requires_grad = False

    # ==========================================
    # 3. Training Setup
    # ==========================================
    BATCH_SIZE = 64
    EPOCHS = 50
    LR = 1e-4
    EMA_RATE = 0.95 # How fast the Target network tracks the Student

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    
    optimizer = torch.optim.AdamW(student_unet.parameters(), lr=LR)

    # ==========================================
    # 4. Consistency Distillation Loop
    # ==========================================
    print("\nStarting Distillation...")
    for epoch in range(EPOCHS):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for imgs, _ in pbar:
            imgs = imgs.to(device)
            B = imgs.shape[0]

            # 1. Pick a random timestep block (t_{n+1} and t_n)
            # We avoid the absolute last step (0) to prevent index out-of-bounds
            step_indices = torch.randint(0, len(timesteps) - 1, (B,), device=device)
            
            t_n_plus_1 = timesteps[step_indices] # The noisier step
            t_n = timesteps[step_indices + 1]    # One step closer to clean image

            # 2. Add noise to real images to get x_{t_{n+1}}
            # --- THE FIX: Cache the alphas table on the GPU ---
            alphas_cumprod_device = scheduler.alphas_cumprod.to(device)

            # 2. Add noise to real images to get x_{t_{n+1}}
            noise = torch.randn_like(imgs)
            alphas_cumprod_n_plus_1 = alphas_cumprod_device[t_n_plus_1].view(-1, 1, 1, 1)
            x_t_n_plus_1 = torch.sqrt(alphas_cumprod_n_plus_1) * imgs + torch.sqrt(1 - alphas_cumprod_n_plus_1) * noise

            # 3. Use Teacher to take ONE step backward to get x_{t_n}
            with torch.no_grad():
                teacher_noise_pred = teacher_unet(x_t_n_plus_1, t_n_plus_1).sample
                
                # Standard DDIM step equation using the GPU-cached table
                alpha_prod_t = alphas_cumprod_device[t_n_plus_1].view(-1, 1, 1, 1)
                alpha_prod_t_prev = alphas_cumprod_device[t_n].view(-1, 1, 1, 1)
                
                pred_x0_teacher = (x_t_n_plus_1 - torch.sqrt(1 - alpha_prod_t) * teacher_noise_pred) / torch.sqrt(alpha_prod_t)
                dir_xt_teacher = torch.sqrt(1 - alpha_prod_t_prev) * teacher_noise_pred
                
                # The Teacher's estimate of x_{t_n}
                x_t_n = torch.sqrt(alpha_prod_t_prev) * pred_x0_teacher + dir_xt_teacher

            # 4. Student predicts the final image x0 directly from x_{t_{n+1}}
            student_noise_pred = student_unet(x_t_n_plus_1, t_n_plus_1).sample
            student_pred_x0 = get_predicted_x0(student_unet, scheduler, x_t_n_plus_1, t_n_plus_1, student_noise_pred)

            # 5. Target Network predicts the final image x0 from x_{t_n}
            with torch.no_grad():
                target_noise_pred = target_unet(x_t_n, t_n).sample
                target_pred_x0 = get_predicted_x0(target_unet, scheduler, x_t_n, t_n, target_noise_pred)

            # 6. Consistency Loss: The Student's guess must match the Target's guess
            # Note: We use Huber loss (smooth_l1) because it is more stable for distillation
            loss = F.smooth_l1_loss(student_pred_x0, target_pred_x0)

            # 7. Backprop & Update Student
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 8. Exponential Moving Average (EMA) Update for Target Network
            with torch.no_grad():
                for param_student, param_target in zip(student_unet.parameters(), target_unet.parameters()):
                    param_target.data.mul_(EMA_RATE).add_(param_student.data, alpha=1 - EMA_RATE)

            pbar.set_postfix({'Loss': f"{loss.item():.4f}"})

        # Save Checkpoint
        torch.save(student_unet.state_dict(), f"student_unet_epoch_{epoch+1}.pt")
        print(f"Saved Student Checkpoint for Epoch {epoch+1}")

if __name__ == "__main__":
    train_consistency_distillation()
    
    
# Epoch 1/50: [09:52<00:00,  1.32it/s, Loss=0.0095]
# Epoch 2/50: [09:52<00:00,  1.32it/s, Loss=0.0003]
# Epoch 3/50: [09:51<00:00,  1.32it/s, Loss=0.0003]
# Epoch 4/50: [09:51<00:00,  1.32it/s, Loss=0.0013]
# Epoch 5/50: [09:51<00:00,  1.32it/s, Loss=0.0009]
# Epoch 6/50: [09:51<00:00,  1.32it/s, Loss=0.0001]
# Epoch 7/50: [09:51<00:00,  1.32it/s, Loss=0.0002]
# Epoch 8/50: [09:51<00:00,  1.32it/s, Loss=0.0002]
# Epoch 9/50: [09:51<00:00,  1.32it/s, Loss=0.0001]
# Epoch 10/50: [09:52<00:00,  1.32it/s, Loss=0.0008]
# Epoch 11/50: [09:50<00:00,  1.32it/s, Loss=0.0001]
# Epoch 12/50: [09:54<00:00,  1.31it/s, Loss=0.0008]
# Epoch 13/50: [09:57<00:00,  1.31it/s, Loss=0.0002]
# Epoch 14/50: [09:50<00:00,  1.32it/s, Loss=0.0003]
# Epoch 15/50: [09:48<00:00,  1.33it/s, Loss=0.0003]
# Epoch 16/50: [09:47<00:00,  1.33it/s, Loss=0.0006]
# Epoch 17/50: [09:48<00:00,  1.33it/s, Loss=0.0001]
# Epoch 18/50: [09:47<00:00,  1.33it/s, Loss=0.0002]
# Epoch 19/50: [09:47<00:00,  1.33it/s, Loss=0.0002]
# Epoch 20/50: [09:46<00:00,  1.33it/s, Loss=0.0001]
# Epoch 21/50: [09:46<00:00,  1.33it/s, Loss=0.0002]
# Epoch 22/50: [09:46<00:00,  1.33it/s, Loss=0.0001]
# Epoch 23/50: [09:47<00:00,  1.33it/s, Loss=0.0002]
# Epoch 24/50: [09:46<00:00,  1.33it/s, Loss=0.0003]
# Epoch 25/50: [09:46<00:00,  1.33it/s, Loss=0.0001]
# Epoch 26/50: [09:46<00:00,  1.33it/s, Loss=0.0023]
# Epoch 27/50: [09:46<00:00,  1.33it/s, Loss=0.0003]
# Epoch 28/50: [09:46<00:00,  1.33it/s, Loss=0.0001]
# Epoch 29/50: [09:47<00:00,  1.33it/s, Loss=0.0001]
# Epoch 30/50: [09:47<00:00,  1.33it/s, Loss=0.0015]
# Epoch 31/50: [09:46<00:00,  1.33it/s, Loss=0.0001]
# Epoch 32/50: [09:46<00:00,  1.33it/s, Loss=0.0024]
# Epoch 33/50: [09:46<00:00,  1.33it/s, Loss=0.0003]
# Epoch 34/50: [09:46<00:00,  1.33it/s, Loss=0.0001]
# Epoch 35/50: [09:46<00:00,  1.33it/s, Loss=0.0002]
# Epoch 36/50: [09:46<00:00,  1.33it/s, Loss=0.0002]
# Epoch 37/50: [09:46<00:00,  1.33it/s, Loss=0.0001]
# Epoch 38/50: [09:46<00:00,  1.33it/s, Loss=0.0011]
# Epoch 39/50: [09:46<00:00,  1.33it/s, Loss=0.0002]
# Epoch 40/50: [09:46<00:00,  1.33it/s, Loss=0.0002]
# Epoch 41/50: [09:46<00:00,  1.33it/s, Loss=0.0057]
# Epoch 42/50: [09:46<00:00,  1.33it/s, Loss=0.0002]
# Epoch 43/50: [09:46<00:00,  1.33it/s, Loss=0.0009]
# Epoch 44/50: [09:47<00:00,  1.33it/s, Loss=0.0002]
# Epoch 45/50: [09:49<00:00,  1.33it/s, Loss=0.0001]
# Epoch 46/50: [09:49<00:00,  1.33it/s, Loss=0.0002]
# Epoch 47/50: [09:49<00:00,  1.33it/s, Loss=0.0002]
# Epoch 48/50: [09:49<00:00,  1.33it/s, Loss=0.0001]
# Epoch 49/50: [09:47<00:00,  1.33it/s, Loss=0.0002]
# Epoch 50/50: [09:48<00:00,  1.33it/s, Loss=0.0001]
