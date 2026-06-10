## Lightweight Diffusion Models <br><sub>Accelerating Inference for Resource-Constrained Environments</sub>

- image of a batch of generated images on CIFAR-10

## Overview

Denoising Diffusion Probabilistic Models (DDPMs) [3] achieve state-of-the-art results in image
generation, but their iterative sampling process is notoriously slow and computationally expensive.

This project investigates methods for accelerating inference in diffusion models trained on CIFAR-10 while preserving generation quality. In particular, it focuses on Progressive Distillation, a technique that transfers the behavior of a multi-step diffusion sampler into a model requiring significantly fewer inference steps.

The implementation and experiments in this repository are primarily based on the following works:

- **DDPM** [3]: foundational diffusion model formulation.
- **DDIM** [2]: deterministic and efficient sampling procedure.
- **Progressive Distillation** [1]: iterative reduction of sampling steps through knowledge distillation.

The goal is to analyze the trade-off between inference speed and sample quality as the number of denoising steps is progressively reduced.

## Methodology

While multiple approaches have been tested, we ultimately reached the highest performance when using a combination of Denoising Diffusion Implicit Models (DDIM) [2] and Progressive Distillation [1]. This work focuses mainly on the combination of the latter, but all the approaches implemented can be found in Appendix A.2.

More concretely, we started off with the official Google implementation of the DDPM paper [4], which uses 1000 sampling steps, and built a DDIM sampling procedure on top of it. This allowed us to reduce the number of steps to roughly 25–30 without retraining the model and with only a minor sacrifice in performance. Once we achieved that, we iteratively reduced the number of steps through Progressive Distillation, first training a 25-step student model on the base DDPM with DDIM sampling, and then moving down to 12 and 8 steps (see Figure 1). Our fastest model achieves a [TO INSERT THE SPEEDUP] speedup during inference, while maintaining an FID of 15.9995 and an IS of 8.6021. A detailed description of how the IS and FID scores are computed can be found in Appendix A.1.

![Pipeline](docs/diagram.png)

*Figure 1: Diffusion model acceleration pipeline*


## 4. Results

To evaluate the results obtained, we conducted two different studies. The first one is a Downstream Impact Evaluation (Section 4.1), where we quantify the speed-up factor and compare the visual quality and quantitative metrics of the accelerated model against the baseline. The second is a Fidelity vs. Diversity study (Section 4.2), where we conduct an ablation study on the number of sampling steps for each method. There, we analyze how the proposed efficiency method handles severe step reductions compared to the standard DDPM schedule, i.e., we evaluate its behavior under stress.

### 4.1 Downstream Impact Evaluation
> Downstream Impact Evaluation: Quantify the speed-up factor and compare the visual quality and quantitative
metrics of the accelerated model against the baseline.

For this experiment, we evaluate the speed-up factor of different setups across varying batch sizes using an NVIDIA RTX 5000 Ada Generation GPU. Detailed results for all batch sizes are provided in Appendix A.3. In subsequent experiments, the speed-up factor is defined relative to a batch size of 32.

After quantifying the speed-up factor, we compute the FID and IS for each model and compare the results (see Table 2).


| Method | Steps | Speed-up factor | FID | IS |
|--------|-------|-----------------|-----|-----|
| **DDPM** | 1000 | x1 | - | - |
| **DDIM** | 25 | - | 16.3709 | 8.1425 ± 0.2514 |
| **DDIM + Progressive Distillation** | 25 | - | 14.1555 | 8.3367 ± 0.3336 |
| | 12 | - | 12.9952 | 8.4135 ± 0.2914 |
| | 8 | - | 15.9995 | 8.6021 ± 0.4500 |

*Table 2: Speed-up factor and quantitative metrics comparison*

> analyze the results!!!!


As for the visual quality

as for the visual quality, ...

- a figure with some pictures for each number of steps




### 4.2 Fidelity vs. Diversity Study

- DDIM as the sampler + progressive distillation to compensate for quality loss at fewer steps


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

[4] Google Research. "DDPM CIFAR-10 32x32." Hugging Face Model Hub.
https://huggingface.co/google/ddpm-cifar10-32

----
TO BE DELETED IF NOT MENTIONED

[2] Song, Y., Dhariwal, P., Chen, M., & Sutskever, I. "Consistency Models." ICML 2023.

[3] Zhou, M., Zheng, H., Wang, Z., Yin, M., & Huang, H. "Score Identity Distillation: Exponentially Fast Distillation of Pretrained Diffusion Models for One-Step Generation." ICML 2024.

[4] Lu, C., & Song, Y. "Simplifying, Stabilizing and Scaling Continuous-Time Consistency Models." arXiv 2024.

[5] Karras, T., Aittala, M., Aila, T., & Laine, S. "Elucidating the Design Space of Diffusion-Based Generative Models." NeurIPS 2022.

## Abstract
### A.1 FID and IS scores
- evaluation methods specifics (how are FID and IS computed)
- flaws and differences to other standards?

### A.2 Other approaches
- describe other approaches
- considered other approachers (e.g latent space), but since our data (CIFAR-10) is so small it did not make sense

### A.3 Inference-Time Benchmark
For the evaluation of inference time for each model, we perform 10 warm-up runs followed by 50 measured runs per configuration. After collecting the results, we compute the median execution time across the 50 measured runs, as well as the milliseconds per image and images per second metrics (see Table X). All experiments are conducted on a single NVIDIA RTX 5000 Ada Generation GPU.

| model       | steps | batch | median (ms) | ms/img  | img/s |
|-------------|-------|-------|-------------|---------|--------|
| **DDIM + Progressive Distillation**  | 25    | 1     | 111.22      | 111.219 | 9.0    |
|   |     | 2     | 113.54      | 56.768  | 17.6   |
|   |     | 4     | 114.64      | 28.659  | 34.9   |
|   |     | 8     | 113.12      | 14.140  | 70.7   |
|   |     | 16    | 172.97      | 10.811  | 92.5   |
|   |     | 32    | 310.40      | 9.700   | 103.1  |
|   |      | 64    | 697.32      | 10.896  | 91.8   |
| **DDIM + Progressive Distillation**  | 12    | 1     | 53.74       | 53.738  | 18.6   |
|   |     | 2     | 54.31       | 27.157  | 36.8   |
|   |    | 4     | 54.89       | 13.723  | 72.9   |
|  |   | 8     | 54.44       | 6.805   | 146.9  |
|   |   | 16    | 85.30       | 5.331   | 187.6  |
|   |    | 32    | 149.75      | 4.680   | 213.7  |
|  |   | 64    | 315.03      | 4.922   | 203.2  |
| **DDIM + Progressive Distillation**    | 8     | 1     | 35.86       | 35.860  | 27.9   |
|  |     | 2     | 36.89       | 18.443  | 54.2   |
|  |    | 4     | 36.94       | 9.235   | 108.3  |
|  |  | 8     | 36.95       | 4.619   | 216.5  |
|  |    | 16    | 57.56       | 3.598   | 278.0  |
|  |    | 32    | 101.10      | 3.159   | 316.5  |
|  |     | 64    | 212.76      | 3.324   | 300.8  |
*Table X: Diffusion model acceleration pipeline*

> Change table name here and in the text that references it

As observed, execution benefits from parallelism within the GPU, leading to a progressive increase in throughput (images/s), which peaks at a batch size of 32.