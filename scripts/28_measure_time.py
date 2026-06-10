import torch
import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from diffusers import DDPMPipeline, DDIMScheduler, UNet2DModel

def benchmark_inference():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = "fast_professor_8step.pt"
    num_images = 100
    inference_steps = 8

    # 1. Load Model & Scheduler
    print(f"Loading model from {model_path}...")
    pipe = DDPMPipeline.from_pretrained("google/ddpm-cifar10-32").to(device)
    model = UNet2DModel.from_config(pipe.unet.config).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    scheduler.set_timesteps(inference_steps)
    timesteps = scheduler.timesteps.to(device)
    alphas = scheduler.alphas_cumprod.to(device)

    latencies = []

    print(f"Starting benchmark (Generating {num_images} images individually)...")

    # Warm-up (Gives the GPU a moment to spin up and cache kernels)
    with torch.no_grad():
        dummy_x = torch.randn(1, 3, 32, 32, device=device)
        for t in timesteps:
            model(dummy_x, t.unsqueeze(0)).sample

    # 2. Main Timing Loop
    with torch.no_grad():
        for _ in tqdm(range(num_images)):
            # Synchronization events for precise GPU timing
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            x = torch.randn(1, 3, 32, 32, device=device)

            start_event.record()
            
            # The 8-Step Sampling Loop
            for i, t in enumerate(timesteps):
                t_batch = t.unsqueeze(0)
                noise_pred = model(x, t_batch).sample
                
                a_s = alphas[t].view(-1, 1, 1, 1)
                if i == len(timesteps) - 1:
                    x = (x - torch.sqrt(1 - a_s) * noise_pred) / torch.sqrt(a_s)
                    break
                    
                next_t = timesteps[i+1]
                a_e = alphas[next_t].view(-1, 1, 1, 1)
                x0_p = (x - torch.sqrt(1 - a_s) * noise_pred) / torch.sqrt(a_s)
                x = torch.sqrt(a_e) * x0_p + torch.sqrt(1 - a_e) * noise_pred
            
            end_event.record()
            
            # Wait for GPU to finish to get the result
            torch.cuda.synchronize()
            latencies.append(start_event.elapsed_time(end_event)) # Returns milliseconds

    # 3. Stats & Distribution
    latencies = np.array(latencies)
    avg_sec = np.mean(latencies) / 1000
    std_sec = np.std(latencies) / 1000

    print("\n" + "="*30)
    print(f"BENCHMARK RESULTS (8 Steps)")
    print(f"Mean Latency: {avg_sec:.4f}s")
    print(f"Std Dev:      {std_sec:.4f}s")
    print(f"Max Latency:  {np.max(latencies)/1000:.4f}s")
    print(f"Min Latency:  {np.min(latencies)/1000:.4f}s")
    print("="*30)

    # Plot Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(latencies / 1000, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title(f"Inference Time Distribution (n={num_images})")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Frequency")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

if __name__ == "__main__":
    benchmark_inference()