## Notation

Each file is tagged with one of the following labels:

- **[Model]** — Defines (and trains) an architecture
- **[Evaluation]** — Script used for evaluating a trained model
- **[Util]** — Shared utilities with no training logic

## Files
1. **[Model]** _01_example_distilled.py_

    A custom VAE compresses images from 32×32 pixel space down to an 8×8 latent space. A teacher UNet is trained in that latent space using a v-prediction objective (predicting the velocity field instead of the raw noise). A student UNet is then trained via progressive distillation: it learns to skip 2 teacher steps in 1 step, halving the required inference steps. It is a three-phase pipeline: train VAE → train teacher → distill student

2. **[Evaluation]** _02_generate_img.py_ 

    Loads the saved VAE + student weights from _01_example_distilled.py_, visualizes VAE reconstruction quality, and measures inference time

3. **[Util]** _03_classes.py_ 

    Shared architecture definitions (no training logic)

4. **[Model]** _04_baseline.py_

    Images are converted to FFT before training (6 channels: real + imaginary parts), so the model learns in frequency space. Architecture: a Diffusion Transformer (DiT) with Linear Attention (O(N) instead of O(N²)) and SwiGLU MLP blocks. Training objective: Rectified Flow (linear interpolation trajectory x_t = t*x1 + (1-t)*x0, target = x1 - x0). Inference needs only 4 Euler steps

5. **[Model]** _05_baseline2.py_ 

    Drops the frequency domain from the previous approach; works directly on RGB pixels. Adds EMA (Exponential Moving Average) for more stable model weights at evaluation time. Adds AMP (Automatic Mixed Precision with float16) for faster GPU training. Adds a OneCycleLR cosine scheduler with 10% linear warmup. Achieved IS ≈ 2.5 after 200 epochs; 30-step Euler sampling

6. **[Utils]** _06_visualize_dataset.py_ 

    Simple CIFAR-10 sample grid visualization

7. **[Evaluation]** _07_eval_model.py_ 

    Large-scale evaluation (10,000 samples) for the model from file 05


8. **[Model]** _08_simplified_baseline.py_

    Replaces Rectified Flow with a DDPM cosine noise schedule (avoids artifacts near t=0/1). Targets the velocity (v), not raw noise, which improves training stability. Adds Min-SNR loss weighting (γ=5): upweights high-noise timesteps to balance the gradient signal. Uses fixed 2D sinusoidal positional embeddings (from the official DiT paper) instead of learned ones. Uses full quadratic attention (scaled_dot_product_attention) with FlashAttention. DDIM sampling at inference (30 steps); best IS ≈ 2.59; evaluated at 50k samples: IS 6.59, FID 0.11.

9. **[Evaluation]** _09_eval_eight.py_
    
    50,000-sample evaluation of the model from file 08

10. **[Model]** _10_overlapping_conv.py_

    Same as 08 but with an Overlapping Conv Stem. Replaces the single large-stride patch embedding (4×4, stride 4) with a 3-layer convolutional stem: 32→16→16→8, using 3×3 kernels, BatchNorm, and GELU. The gradual downsampling captures local edge/texture features before the Transformer sees the patches

11. **[Model]** _11_big_model.py_

    Same as 10 but with a larger model (dim=384 vs 256). Overlapping conv stem + DDPM + v-prediction + DDIM, just scaled up in capacity

12. **[Model]** _12_U_ViT.py_

    The best-performing custom model. Introduces a U-Net style hierarchy into the Vision Transformer.  Architecture: high-res block (16×16 patches) → PatchMerge (4× token reduction to 8×8) → deep transformer mid-blocks → PatchExpand (4× token expansion back to 16×16) → skip connection → high-res output block. Uses dual sinusoidal positional embeddings (one for each resolution). Applies AdaLN-Zero initialization (zero-init of modulation layers) for early training stability. Finer patch size (2×2 instead of 4×4) for richer spatial detail. Trained for 800 epochs; best mini-IS ≈ 5.29 (epoch 200); full evaluation: IS 7.24, FID 27.65

13. **[Model]** _13_huge_model.py_ 

     A large flat DiT for comparison against U-ViT. Pure flat transformer stack (no merge/expand blocks), dim=768 — "ViT-Large" scale. Used to verify whether U-ViT's hierarchy actually helps vs simply having more parameters.

14. **[Model]** _14_google_baseline.py_

    Establishes the baseline. Loads the official HuggingFace google/ddpm-cifar10-32 pretrained model. Compares full DDPM (1000 steps) vs fast DDIM (30 steps) as the quality/speed reference point.

15. **[Model]** _15_teacher_finaly.py_ 
    
    Consistency Distillation from the Google teacher. Student and target networks are both initialized as copies of the Google DDPM teacher. The target network is updated via EMA of the student. The student is trained to predict the clean image x0 directly from a noisy x_t, skipping the teacher's multi-step trajectory — enabling 1-step generation

16. **[Evaluation]** _16_teacher_vs_student_benchmark.py_

    Benchmarking script. Loads both the Google DDPM teacher pipeline and the trained consistency student. Computes FID, IS, and generation time for both, side-by-side
