## Lightweight Diffusion Models <br><sub>Accelerating Training/Inference for Resource-Constrained Environments</sub>


## Abstract

## Introduction

## Related Works
- foundational papers: [5,7], https://huggingface.co/google/ddpm-cifar10-32
- distillation techniques: [1, 2, 3, 4]
- sampling algorithms: [6]

## Proposed Methods
> Anatomical Guidance Integration: Formulate and integrate an algorithmic improvement (e.g., DDIM sampling
strategy, a latent-space approach, progressive distillation, or something new!) aimed at accelerating inference
or training.

1. Progressive Distillation 

## Results
> Downstream Impact Evaluation: Quantify the speed-up factor and compare the visual quality and quantitative
metrics of the accelerated model against the baseline.

> Fidelity vs. Diversity Study : Conduct an ablation study on the number of sampling steps (e.g., T = 1000 vs.
T = 100 vs. T = 10). Analyze how the proposed efficiency method handles severe step reductions compared
to the standard DDPM scheduler.
## Conclusion

## References
[1] Salimans, T., & Ho, J. ”Progressive Distillation for Fast Sampling of Diffusion Models.” ICLR 2022.

[2] Song, Y., Dhariwal, P., Chen, M., & Sutskever, I. "Consistency Models." ICML 2023.

[3] Zhou, M., Zheng, H., Wang, Z., Yin, M., & Huang, H. "Score Identity Distillation: Exponentially Fast Distillation of Pretrained Diffusion Models for One-Step Generation." ICML 2024.

[4] Lu, C., & Song, Y. "Simplifying, Stabilizing and Scaling Continuous-Time Consistency Models." arXiv 2024.

[5] Karras, T., Aittala, M., Aila, T., & Laine, S. "Elucidating the Design Space of Diffusion-Based Generative Models." NeurIPS 2022.

[6] Song, J., Meng, C., & Ermon, S. ”Denoising Diffusion Implicit Models.” ICLR 2021.

[7] Ho, J., Jain, A., & Abbeel, P. "Denoising Diffusion Probabilistic Models." NeurIPS 2020.