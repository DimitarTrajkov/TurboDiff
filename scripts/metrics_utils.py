"""
metrics_utils.py

Shared evaluation logic for the generative metrics used across scripts:
FID, Inception Score, and improved Precision/Recall (Kynkaanniemi et al. 2019).

FID and IS use the same torchmetrics setup as ddpm_arch.run_eval (feature=2048,
normalize=True). Precision/Recall are computed on the same Inception-2048 features
FID uses, so all four metrics live in one feature space.
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore


# ─────────────────────────────────────────────────────────────
# Real data
# ─────────────────────────────────────────────────────────────
def load_real_images(num_samples, data_root="./data"):
    """First num_samples CIFAR-10 train images as a CPU tensor in [0, 1]."""
    dataset = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=transforms.ToTensor())
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=2)
    imgs, seen = [], 0
    for x, _ in loader:
        imgs.append(x)
        seen += x.shape[0]
        if seen >= num_samples:
            break
    return torch.cat(imgs, dim=0)[:num_samples]


def make_extractor(device):
    """The Inception-2048 feature extractor that FID uses (for Precision/Recall features)."""
    return FrechetInceptionDistance(feature=2048, normalize=True).to(device).inception


# ─────────────────────────────────────────────────────────────
# Inception features + improved Precision / Recall
# ─────────────────────────────────────────────────────────────
@torch.no_grad()
def inception_features(extractor, imgs01, device, chunk=250):
    """Extract Inception-2048 features for images in [0, 1]."""
    feats = []
    for i in range(0, imgs01.shape[0], chunk):
        batch = imgs01[i:i + chunk].to(device)
        feats.append(extractor((batch * 255).byte()).float())
    return torch.cat(feats, dim=0)


@torch.no_grad()
def _kth_nn_radius(feats, k, chunk=2000):
    """Distance from each point to its k-th nearest neighbour within the same set."""
    n = feats.shape[0]
    radii = torch.empty(n, device=feats.device)
    for i in range(0, n, chunk):
        d = torch.cdist(feats[i:i + chunk], feats)           # (c, n)
        # exclude self (distance 0): the (k+1)-th smallest is the k-th neighbour
        radii[i:i + chunk] = torch.topk(d, k + 1, largest=False).values[:, -1]
    return radii


@torch.no_grad()
def _fraction_within(query, ref, ref_radii, chunk=2000):
    """Fraction of query points lying inside ANY reference hypersphere."""
    n = query.shape[0]
    inside = torch.zeros(n, dtype=torch.bool, device=query.device)
    for i in range(0, n, chunk):
        d = torch.cdist(query[i:i + chunk], ref)             # (c, m)
        inside[i:i + chunk] = (d <= ref_radii.unsqueeze(0)).any(dim=1)
    return inside.float().mean().item()


@torch.no_grad()
def precision_recall(real_feats, fake_feats, k=3):
    """
    Improved precision & recall (Kynkaanniemi et al. 2019).
      precision = fraction of fakes inside the real feature manifold
      recall    = fraction of reals inside the fake feature manifold
    """
    real_radii = _kth_nn_radius(real_feats, k)
    fake_radii = _kth_nn_radius(fake_feats, k)
    precision = _fraction_within(fake_feats, real_feats, real_radii)
    recall = _fraction_within(real_feats, fake_feats, fake_radii)
    return precision, recall


# ─────────────────────────────────────────────────────────────
# Setup + scoring
# ─────────────────────────────────────────────────────────────
def prepare_real(num_samples, data_root, device):
    """Load real images and their Inception features once (reused across step counts)."""
    real_imgs01 = load_real_images(num_samples, data_root)
    extractor = make_extractor(device)
    real_feats = inception_features(extractor, real_imgs01, device)
    return extractor, real_imgs01, real_feats


@torch.no_grad()
def compute_metrics(fake_uint8, real_imgs01, real_feats, extractor, device, batch, knn):
    """FID / IS / Precision / Recall for a set of fake images (uint8, NCHW)."""
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    is_metric = InceptionScore(normalize=True).to(device)

    for i in range(0, real_imgs01.shape[0], batch):
        fid.update(real_imgs01[i:i + batch].to(device), real=True)

    fake_feats = []
    for i in range(0, fake_uint8.shape[0], batch):
        fake01 = fake_uint8[i:i + batch].to(device).float() / 255.0
        fid.update(fake01, real=False)
        is_metric.update(fake01)
        fake_feats.append(inception_features(extractor, fake01, device))
    fake_feats = torch.cat(fake_feats, dim=0)

    precision, recall = precision_recall(real_feats, fake_feats, k=knn)
    is_mean, is_std = is_metric.compute()
    return {"fid": fid.compute().item(), "is_mean": is_mean.item(),
            "is_std": is_std.item(), "precision": precision, "recall": recall}
