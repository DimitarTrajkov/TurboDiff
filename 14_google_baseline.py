import torch
from diffusers import DDPMPipeline, DDIMScheduler
from torchvision.utils import make_grid
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import time

def save_image_grid(pil_images, filename):
    """Helper function to stitch 64 PIL images into an 8x8 grid and save it."""
    # Convert PIL images to PyTorch tensors
    tensors = [ToTensor()(img) for img in pil_images]
    batch_tensor = torch.stack(tensors)
    
    # Create an 8x8 grid
    grid = make_grid(batch_tensor, nrow=8, padding=2, normalize=False)
    grid_np = grid.permute(1, 2, 0).cpu().numpy()
    
    # Save to disk
    plt.imsave(filename, grid_np)
    print(f" => Successfully saved to: {filename}")

def run_comparison():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    # Load the official baseline pipeline
    print("Loading 'google/ddpm-cifar10-32'...")
    pipeline = DDPMPipeline.from_pretrained("google/ddpm-cifar10-32").to(device)

    BATCH_SIZE = 64
    SEED = 42 # Fixed seed so both methods start from the exact same noise

    print("\n" + "="*40)
    print(" TEST 1: ORIGINAL DDPM (1000 STEPS)")
    print("="*40)
    print("Generating... (This will take 10-20 seconds)")
    
    generator = torch.Generator(device=device).manual_seed(SEED)
    
    start_time = time.time()
    # By default, this pipeline uses 1000 steps and the DDPM scheduler
    ddpm_output = pipeline(batch_size=BATCH_SIZE, generator=generator)
    ddpm_time = time.time() - start_time
    
    print(f"Generation Time: {ddpm_time:.2f} seconds")
    save_image_grid(ddpm_output.images, "1_google_ddpm_1000_steps.png")

    print("\n" + "="*40)
    print(" TEST 2: FAST DDIM (30 STEPS)")
    print("="*40)
    print("Swapping scheduler and generating... Watch how fast this is!")
    
    # Swap the scheduler mathematically
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    
    # Reset the generator seed to 42 so we get the exact same starting noise
    generator = torch.Generator(device=device).manual_seed(SEED)
    
    start_time = time.time()
    # Now we explicitly tell it to only take 30 steps
    ddim_output = pipeline(
        batch_size=BATCH_SIZE, 
        num_inference_steps=30, 
        generator=generator
    )
    ddim_time = time.time() - start_time
    
    print(f"Generation Time: {ddim_time:.2f} seconds")
    save_image_grid(ddim_output.images, "2_google_ddim_30_steps.png")

    print("\n" + "="*40)
    print(" SUMMARY")
    print("="*40)
    print(f"DDPM Time: {ddpm_time:.2f}s")
    print(f"DDIM Time: {ddim_time:.2f}s")
    print(f"Speedup:   {ddpm_time / ddim_time:.1f}x faster!")
    print("Check your folder to compare the image quality!")

if __name__ == "__main__":
    run_comparison()