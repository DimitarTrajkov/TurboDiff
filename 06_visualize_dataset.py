import matplotlib.pyplot as plt
from torchvision.utils import make_grid


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


BATCH_SIZE = 128

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
])
dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
    
# Get one batch
imgs, _ = next(iter(dataloader))  # uses your CIFAR-10 dataloader
imgs = imgs[:32]  # take first 32 images

# Make grid (8x4)
grid = make_grid(imgs, nrow=8, normalize=True, value_range=(-1, 1))

# Convert to numpy for plotting
grid_np = grid.permute(1, 2, 0).cpu().numpy()

# Plot
plt.figure(figsize=(8, 8))
plt.imshow(grid_np)
plt.title("32 CIFAR-10 Samples")
plt.axis('off')
plt.show()