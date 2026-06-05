## Lightweight Diffusion Models <br><sub>Accelerating Training/Inference for Resource-Constrained Environments</sub>


## Abstract

## 1. Introduction

## 2. Related Works
- foundational papers: [3], https://huggingface.co/google/ddpm-cifar10-32
- distillation techniques: [1]
- sampling algorithms: [2]

## 3. Proposed Methods
> Anatomical Guidance Integration: Formulate and integrate an algorithmic improvement (e.g., DDIM sampling
strategy, a latent-space approach, progressive distillation, or something new!) aimed at accelerating inference
or training.

The original model: DDPM, 1000 steps
After DDIM swap (free, no training): same weights, ~25–30 steps, quality slightly drops
After progressive distillation (fine-tuning): same architecture, 25 → 12 → 8 steps, quality is partially recovered because the model is now trained to be accurate at those specific step counts

- DDIM as the sampler + progressive distillation to compensate for quality loss at fewer steps
- DDIM
- progressive distillation
- considered other approachers (e.g latent space), but since our data (CIFAR-10) is so small it did not make sense

1. Progressive Distillation 

## 4. Results

### 4.1 Downstream Impact Evaluation
> Downstream Impact Evaluation: Quantify the speed-up factor and compare the visual quality and quantitative
metrics of the accelerated model against the baseline.

- question we want to address: Quantify the speed-up factor and compare the visual quality and quantitative
metrics of the accelerated model against the baseline.


| Method | Steps | Speed-up factor | FID | IS |
|--------|-------|-----------------|-----|-----|
| **DDPM** | 1000 | x1 | - | - |
| **DDIM** | 25 | - | - | - |
| **DDIM + Progressive Distillation** | 25 | - | - | - |
| | 12 | - | - | - |
| | 8 | - | - | - |




- a figure with some pictures for each number of steps
- WITHOUT DDIM!!! just base DDPM




### 4.2 Fidelity vs. Diversity Study


> Fidelity vs. Diversity Study : Conduct an ablation study on the number of sampling steps (e.g., T = 1000 vs.
T = 100 vs. T = 10). Analyze how the proposed efficiency method handles severe step reductions compared
to the standard DDPM scheduler.

- question to address:  Analyze how the proposed efficiency method handles severe step reductions compared
to the standard DDPM scheduler.

The point is to show the shape of the degradation curve: standard DDPM falls apart (loses fidelity, or collapses diversity) when starved of steps, while your method stays flat. It's diagnostic—an ablation that explains the behavior under stress, specifically split along the fidelity/diversity axis (do samples stay sharp? do they stay varied?).  The expected and interesting result is that they fail differently. It's about characterizing behavior under stress and decomposing that behavior into the two things practitioners actually care about, so you can say not just "my method survives aggressive step reduction" but specifically "it preserves diversity that standard DDPM loses" (or fidelity, or both)—a much sharper and more defensible scientific claim.

> | FID ↓ | IS ↑ | Precision ↑ (fidelity) | Recall ↑ (diversity) |

| Method | Steps | FID | IS | Precision | Recall |
|-------|-------|---------|-------|----------------|----------------|
| **DDPM** | 1000 | - | - | - | -|
| | 100 | - | - | - | - |
| | 8 | - | - | - | - |
| **DDIM** | 25 | - | - | - | - |
| | 12 | - | - | - | - |
| | 8 | - | - | - | - |
| **DDIM + Progressive Distillation** | 25 | - | - | - | - |
| | 12 | - | - | - | - |
| | 8 | - | - | -| - |




## Conclusion

## References
[1] Salimans, T., & Ho, J. ”Progressive Distillation for Fast Sampling of Diffusion Models.” ICLR 2022.

[2] Song, J., Meng, C., & Ermon, S. ”Denoising Diffusion Implicit Models.” ICLR 2021.

[3] Ho, J., Jain, A., & Abbeel, P. "Denoising Diffusion Probabilistic Models." NeurIPS 2020.

----
TO BE DELETED IF NOT MENTIONED

[2] Song, Y., Dhariwal, P., Chen, M., & Sutskever, I. "Consistency Models." ICML 2023.

[3] Zhou, M., Zheng, H., Wang, Z., Yin, M., & Huang, H. "Score Identity Distillation: Exponentially Fast Distillation of Pretrained Diffusion Models for One-Step Generation." ICML 2024.

[4] Lu, C., & Song, Y. "Simplifying, Stabilizing and Scaling Continuous-Time Consistency Models." arXiv 2024.

[5] Karras, T., Aittala, M., Aila, T., & Laine, S. "Elucidating the Design Space of Diffusion-Based Generative Models." NeurIPS 2022.

