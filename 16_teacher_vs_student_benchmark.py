import torch
import time
from tqdm import tqdm
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from diffusers import DDPMPipeline, DDIMScheduler
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. Generation Helper Functions
# ==========================================

@torch.no_grad()
def generate_diffusers_pipeline(pipeline, batch_size, steps, device):
    """Generates a batch using standard HuggingFace pipelines."""
    start_time = time.time()
    
    # Generate the images
    out = pipeline(batch_size=batch_size, num_inference_steps=steps, output_type="pt")
    gen_time = time.time() - start_time
    
    images = out.images
    
    # Catch the NumPy array fallback
    if isinstance(images, np.ndarray):
        fake_imgs = torch.from_numpy(images).to(device)
        # NumPy outputs are usually (B, H, W, C). We need (B, C, H, W)
        if fake_imgs.shape[-1] == 3:
            fake_imgs = fake_imgs.permute(0, 3, 1, 2)
    else:
        # If it correctly returned a tensor
        fake_imgs = images.to(device)
        
    # Convert [0, 1] to [-1, 1] to match your evaluation math
    fake_imgs = fake_imgs * 2.0 - 1.0 
    
    return fake_imgs, gen_time




@torch.no_grad()
def generate_consistency_student(unet, scheduler, batch_size, device):
    """Generates a batch in exactly 1 step using your Consistency Student."""
    start_time = time.time()
    
    # Start from pure noise
    noise = torch.randn(batch_size, 3, 32, 32, device=device)
    
    # Use the absolute highest timestep (e.g., 999)
    t = torch.tensor([scheduler.config.num_train_timesteps - 1] * batch_size, device=device)
    
    # Predict the noise
    noise_pred = unet(noise, t).sample
    
    # Apply your get_predicted_x0 math to jump straight to the final image
    alphas_cumprod_device = scheduler.alphas_cumprod.to(device)
    alpha_prod_t = alphas_cumprod_device[t].view(-1, 1, 1, 1)
    beta_prod_t = 1 - alpha_prod_t
    
    pred_x0 = (noise - torch.sqrt(beta_prod_t) * noise_pred) / torch.sqrt(alpha_prod_t)
    
    gen_time = time.time() - start_time
    
    # pred_x0 is naturally in [-1, 1]
    return pred_x0.clamp(-1, 1), gen_time

@torch.no_grad()
def generate_consistency_student_multistep(unet, scheduler, batch_size, device, steps=5):
    """Generates a batch using Multi-Step Consistency Sampling."""
    start_time = time.time()
    
    # Create a schedule of timesteps, e.g., [999, 899, 799, ..., 0]
    # We want to walk backward from high noise to low noise
    timesteps = torch.linspace(scheduler.config.num_train_timesteps - 1, 0, steps, dtype=torch.long, device=device)
    
    # Start from pure noise
    x = torch.randn(batch_size, 3, 32, 32, device=device)
    alphas_cumprod_device = scheduler.alphas_cumprod.to(device)
    
    for i, t in enumerate(timesteps):
        t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
        
        # Predict noise
        noise_pred = unet(x, t_batch).sample
        
        # Predict the clean image (x0)
        alpha_prod_t = alphas_cumprod_device[t_batch].view(-1, 1, 1, 1)
        beta_prod_t = 1 - alpha_prod_t
        pred_x0 = (x - torch.sqrt(beta_prod_t) * noise_pred) / torch.sqrt(alpha_prod_t)
        
        # If it's the final step, we are done! Output the clean image.
        if i == len(timesteps) - 1:
            x = pred_x0
            break
            
        # Otherwise, add noise back to pred_x0 to match the NEXT (cleaner) timestep
        next_t = timesteps[i + 1]
        next_t_batch = torch.full((batch_size,), next_t, device=device, dtype=torch.long)
        
        alpha_prod_t_next = alphas_cumprod_device[next_t_batch].view(-1, 1, 1, 1)
        noise = torch.randn_like(x)
        
        # Re-noise the prediction
        x = torch.sqrt(alpha_prod_t_next) * pred_x0 + torch.sqrt(1 - alpha_prod_t_next) * noise

    gen_time = time.time() - start_time
    
    # pred_x0 is naturally in [-1, 1]
    return x.clamp(-1, 1), gen_time

# ==========================================
# 2. Your Evaluation Loop (Adapted for multiple models)
# ==========================================
@torch.no_grad()
def evaluate_metrics(name, generation_fn, dataloader, device, num_samples=10000, eval_batch_size=256):
    print(f"\n" + "="*50)
    print(f" EVALUATING: {name}")
    print("="*50)
    
    fid = FrechetInceptionDistance(feature=2048).to(device)
    inc = InceptionScore().to(device)

    # STEP 1: Process Real Images in Chunks
    print("Loading Real Images for FID Baseline...")
    real_processed = 0
    couter = num_samples // eval_batch_size + 1
    
    for batch, _ in dataloader:
        if real_processed >= num_samples: break
        
        current_batch_size = batch.shape[0]
        if real_processed + current_batch_size > num_samples:
            batch = batch[:num_samples - real_processed]
            
        batch = batch.to(device)
        real_imgs_uint8 = ((batch + 1.0) * 127.5).clamp(0, 255).byte()
        
        fid.update(real_imgs_uint8, real=True)
        real_processed += batch.shape[0]
        couter -= 1
        print(f"Processed {real_processed}/{num_samples} real images... ({couter} batches left)", end='\r')

    # STEP 2: Process Fake Images in Chunks
    print("\nGenerating Fake Images...")
    fake_processed = 0
    total_gen_time = 0.0
    fake_imgs_for_grid = []
    couter = num_samples // eval_batch_size + 1
    
    while fake_processed < num_samples:
        current_batch_size = min(eval_batch_size, num_samples - fake_processed)
        
        # Call the appropriate generation function
        fake_imgs, gen_time = generation_fn(current_batch_size)
        total_gen_time += gen_time
        
        fake_imgs_uint8 = ((fake_imgs + 1.0) * 127.5).clamp(0, 255).byte()
        
        fid.update(fake_imgs_uint8, real=False)
        inc.update(fake_imgs_uint8)
        
        if len(fake_imgs_for_grid) < 32:
            needed = 32 - len(fake_imgs_for_grid)
            fake_imgs_for_grid.append(fake_imgs[:needed].cpu())
            
        fake_processed += current_batch_size
        couter -= 1
        print(f"Processed {fake_processed}/{num_samples} fake images... ({couter} batches left)", end='\r')

    # STEP 3: Compute Final Scores and Plot
    print("\nComputing Final Metrics (this takes a moment)...")
    fid_score = fid.compute().item()
    is_score_mean, is_score_std = inc.compute()
    
    print(f"--- RESULTS: {name} ---")
    print(f"Total Gen Time:  {total_gen_time:.4f} seconds")
    print(f"Throughput:      {num_samples / total_gen_time:.2f} images/sec")
    print(f"FID Score:       {fid_score:.4f}")
    print(f"Inception Score: {is_score_mean.item():.4f} ± {is_score_std.item():.4f}")

    grid_imgs = torch.cat(fake_imgs_for_grid, dim=0)
    grid = make_grid(grid_imgs, nrow=8, normalize=True, value_range=(-1, 1))
    grid_np = grid.permute(1, 2, 0).numpy()
    
    plt.imshow(grid_np)
    plt.title(f"{name}\nFID: {fid_score:.2f} | IS: {is_score_mean.item():.2f}")
    plt.axis('off')
    
    safe_name = name.replace(" ", "_").replace("(", "").replace(")", "").lower()
    plt.savefig(f"eval_{safe_name}.png", bbox_inches='tight', dpi=150)
    plt.clf() 
    plt.close()
    
    del fid, inc, grid_imgs
    torch.cuda.empty_cache()

# ==========================================
# 3. Main Benchmark Runner
# ==========================================
def run_all_benchmarks():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Setup Dataset for Real Images
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=0)

    # Note: We use 10,000 samples for time's sake. 
    # Standard academic benchmarks use 50,000, but that takes hours on a single GPU.
    NUM_SAMPLES = 10000 
    EVAL_BATCH_SIZE = 128 # Adjusted to prevent OOM on 1000-step DDPM

    # ---------------------------------------------------------
    # TEST 1: Teacher (DDPM 1000 Steps)
    # ---------------------------------------------------------
    teacher_pipe = DDPMPipeline.from_pretrained("google/ddpm-cifar10-32").to(device)
    
    # evaluate_metrics(
    #     name="Teacher (DDPM 1000-Steps)",
    #     generation_fn=lambda b: generate_diffusers_pipeline(teacher_pipe, b, 1000, device),
    #     dataloader=dataloader,
    #     device=device,
    #     num_samples=NUM_SAMPLES,
    #     eval_batch_size=EVAL_BATCH_SIZE
    # )

    # ---------------------------------------------------------
    # TEST 2: Teacher (DDIM 30 Steps)
    # ---------------------------------------------------------
    teacher_pipe.scheduler = DDIMScheduler.from_config(teacher_pipe.scheduler.config)
    
    evaluate_metrics(
        name="Teacher (DDIM 5-Steps)",
        generation_fn=lambda b: generate_diffusers_pipeline(teacher_pipe, b, 5, device),
        dataloader=dataloader,
        device=device,
        num_samples=NUM_SAMPLES,
        eval_batch_size=EVAL_BATCH_SIZE
    )

    # ---------------------------------------------------------
    # TEST 3: Student (Consistency 1 Step)
    # ---------------------------------------------------------
    # print("\nLoading Student Weights...")
    # # Load your trained weights into the UNet
    # student_unet = teacher_pipe.unet
    # # Make sure to point this to your best saved checkpoint
    # student_unet.load_state_dict(torch.load("student_unet_epoch_50.pt", weights_only=True))
    # student_unet.eval()

    # evaluate_metrics(
    #     name="Student (Consistency 1-Step)",
    #     generation_fn=lambda b: generate_consistency_student(student_unet, teacher_pipe.scheduler, b, device),
    #     dataloader=dataloader,
    #     device=device,
    #     num_samples=NUM_SAMPLES,
    #     eval_batch_size=EVAL_BATCH_SIZE
    # )
    
    # evaluate_metrics(
    #     name="Student (Consistency 5-Step) 50 epochs",
    #     generation_fn=lambda b: generate_consistency_student_multistep(student_unet, teacher_pipe.scheduler, b, device, steps=5),
    #     dataloader=dataloader,
    #     device=device,
    #     num_samples=NUM_SAMPLES,
    #     eval_batch_size=EVAL_BATCH_SIZE
    # )

if __name__ == "__main__":
    run_all_benchmarks()
    
    
# --- RESULTS: Teacher (DDIM 30-Steps) ---
# Total Gen Time:  696.4523 seconds
# Throughput:      14.36 images/sec
# FID Score:       17.4852
# Inception Score: 8.0185 ± 0.2179

# epoch 10:
# --- RESULTS: Student (Consistency 1-Step) ---
# Total Gen Time:  54.5914 seconds
# Throughput:      183.18 images/sec
# FID Score:       180.1839
# Inception Score: 2.9810 ± 0.0626

# epoch 50:
# --- RESULTS: Student (Consistency 1-Step) ---
# Total Gen Time:  678.7939 seconds
# Throughput:      14.73 images/sec
# FID Score:       128.3690
# Inception Score: 3.7283 ± 0.0875

# --- RESULTS: Student (Consistency 5-Step) 50 epochs ---
# Total Gen Time:  129.1213 seconds
# Throughput:      77.45 images/sec
# FID Score:       45.7476
# Inception Score: 6.8083 ± 0.1399


# --- RESULTS: Teacher (DDIM 5-Steps) ---
# Total Gen Time:  142.7558 seconds
# Throughput:      70.05 images/sec
# FID Score:       64.7380
# Inception Score: 5.7746 ± 0.1431



# --- RESULTS: Student (Consistency 5-Step) 50 epochs ---
# Total Gen Time:  129.1213 seconds
# Throughput:      77.45 images/sec
# FID Score:       45.7476
# Inception Score: 6.8083 ± 0.1399


# --- RESULTS: Teacher (DDIM 5-Steps) ---
# Total Gen Time:  142.7558 seconds
# Throughput:      70.05 images/sec
# FID Score:       64.7380
# Inception Score: 5.7746 ± 0.1431