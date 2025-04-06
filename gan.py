import os
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
from tqdm import tqdm
import random

# ---------------------
# Configurations
# ---------------------
DATA_PATH = "data/BraTS"
EPOCHS = 100
BATCH_SIZE = 16
IMAGE_SIZE = 64  # Resize slices to 64x64
LATENT_DIM = 100
SAVE_PATH = "outputs"
os.makedirs(SAVE_PATH, exist_ok=True)

# ---------------------
# Data Preparation
# ---------------------
class BraTSDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.images = []

        for filename in os.listdir(data_path):
            if filename.endswith(".nii") or filename.endswith(".nii.gz"):
                path = os.path.join(data_path, filename)
                nii = nib.load(path).get_fdata()
                for i in range(nii.shape[2]):  # 2D slices along axial plane
                    slice_ = nii[:, :, i]
                    if np.max(slice_) > 0:  # Ignore empty slices
                        self.images.append(slice_)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        img = (img - np.min(img)) / (np.max(img) - np.min(img))  # Normalize to [0,1]
        img = np.expand_dims(img, axis=0)  # Add channel
        img = torch.tensor(img, dtype=torch.float32)
        img = transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))(img)
        return img

# ---------------------
# Generator & Discriminator
# ---------------------
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(LATENT_DIM, 256 * 8 * 8),
            nn.ReLU(True),
            nn.Unflatten(1, (256, 8, 8)),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.net(img)

# ---------------------
# Training Setup
# ---------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

dataloader = DataLoader(BraTSDataset(DATA_PATH), batch_size=BATCH_SIZE, shuffle=True)

generator = Generator().to(device)
discriminator = Discriminator().to(device)

criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# ---------------------
# Training Loop
# ---------------------
for epoch in range(EPOCHS):
    for i, real_imgs in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")):
        real_imgs = real_imgs.to(device)

        # Adversarial ground truths
        valid = torch.ones(real_imgs.size(0), 1, device=device)
        fake = torch.zeros(real_imgs.size(0), 1, device=device)

        # ---------------------
        #  Train Generator
        # ---------------------
        optimizer_G.zero_grad()
        z = torch.randn(real_imgs.size(0), LATENT_DIM, device=device)
        gen_imgs = generator(z)
        g_loss = criterion(discriminator(gen_imgs), valid)
        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()
        real_loss = criterion(discriminator(real_imgs), valid)
        fake_loss = criterion(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

    print(f"Epoch {epoch+1}/{EPOCHS} | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

    if (epoch + 1) % 10 == 0:
        torch.save(generator.state_dict(), f"{SAVE_PATH}/generator_epoch{epoch+1}.pth")
        with torch.no_grad():
            sample = generator(torch.randn(16, LATENT_DIM, device=device)).cpu()
            grid = utils.make_grid(sample, nrow=4, normalize=True)
            plt.figure(figsize=(6,6))
            plt.imshow(grid.permute(1, 2, 0).squeeze(), cmap='gray')
            plt.axis('off')
            plt.title(f"Epoch {epoch+1}")
            plt.savefig(f"{SAVE_PATH}/sample_epoch{epoch+1}.png")
            plt.close()

print("Training complete. Models and samples saved to 'outputs/'")



