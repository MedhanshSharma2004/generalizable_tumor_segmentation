# Libraries
import os
import sys
import json
from argparse import Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib.pyplot as plt
import monai
from monai.apps import download_url
from monai.transforms import LoadImage, Orientation, ScaleIntensity
from monai.config import print_config
from monai.utils import set_determinism

from model import DiffuGTS
from latent_inpainting import LatentSpaceInpaintRefiner
from utility import LoadDataset

# Add MAISI scripts folder to Python path
maisi_scripts = os.path.abspath(os.path.join("..", "tutorials", "generation", "maisi", "scripts"))
if maisi_scripts not in sys.path:
    sys.path.insert(0, maisi_scripts)

# Import MAISI utils
from utils import define_instance
from utils_plot import find_label_center_loc, get_xyz_plot, show_image
from diff_model_setting import setup_logging

# Config paths
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR = "path/to/your/dataset"  
BATCH_SIZE = 2
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
NUM_WORKERS = 4
CHECKPOINT_DIR = "checkpoints"
CONFIG_PATH = os.path.abspath(os.path.join(maisi_scripts, "..", "configs", "config_maisi3d-rflow.json"))
VAE_WEIGHTS = os.path.abspath(os.path.join(maisi_scripts, "models", "autoencoder_epoch273.pt"))
DIFFUNET_WEIGHTS = os.path.abspath(os.path.join(maisi_scripts, "models", "diff_unet_3d_rflow.pt"))

RESUME_CKPT = None  # path to resume checkpoint, if any

# Load config
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)
args = Namespace(**config)

# Instantiate VAE
autoencoder = define_instance(args, "autoencoder_def").to(DEVICE)
if VAE_WEIGHTS:
    ckpt = torch.load(VAE_WEIGHTS, map_location=DEVICE)
    if "model" in ckpt:
        ckpt = ckpt["model"]
    autoencoder.load_state_dict(ckpt)
print("VAE loaded and ready.")

# Instantiate Diffusion UNet
diffusion_unet = define_instance(args, "diffusion_unet").to(DEVICE)
ckpt_diff = torch.load(DIFFUNET_WEIGHTS, map_location=DEVICE)
diffusion_unet.load_state_dict(ckpt_diff.get("unet_state_dict", ckpt_diff))
print("Diffusion UNet loaded.")


# Instantiate Noise Scheduler
noise_scheduler = define_instance(args, "noise_scheduler")
mask_generation_noise_scheduler = define_instance(args, "mask_generation_noise_scheduler")

# Dataset and DataLoader
dataset_loader = LoadDataset(DATA_DIR)
dataset_list = dataset_loader.load_dataset()  # returns list of dicts
class Simple3DDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_list):
        self.dataset_list = dataset_list

    def __len__(self):
        return len(self.dataset_list)

    def __getitem__(self, idx):
        entry = self.dataset_list[idx]
        image = entry['image']          # tensor (1, D, H, W)
        seg_mask = entry['label']       # segmentation mask
        embeddings = entry['embeddings']  # text embeddings tensor
        class_label = entry.get('class_label', None)  # image-level class label
        return image, class_label, embeddings, seg_mask

train_dataset = Simple3DDataset(dataset_list)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

# Latent-space inpainting refiner
latent_refiner = LatentSpaceInpaintRefiner(
    autoencoder=autoencoder,
    diffusion_unet=diffusion_unet,
    noise_scheduler=noise_scheduler,
    device=DEVICE,
)

# Instantiate DiffuGTS
model = DiffuGTS(
    autoencoder=autoencoder,
    latent_inpaint_refiner=latent_refiner,
    alpha=0.1,
    device=DEVICE
)
print("DiffuGTS ready for training!")

# Optimizer & Scheduler
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader)*NUM_EPOCHS)

# Training Loop
for epoch in range(NUM_EPOCHS):
    model.train()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
    epoch_loss = 0.0

    for batch in pbar:
        images, labels, text_embeddings, seg_masks = batch
        images = images.to(DEVICE).float()
        text_embeddings = text_embeddings.to(DEVICE).float()
        labels = labels.to(DEVICE).long() if labels is not None else None
        if seg_masks is not None:
            seg_masks = seg_masks.to(DEVICE).float()

        optimizer.zero_grad()

        # Forward pass
        out = model(images, text_embeddings, class_labels=labels)

        # Compute combined loss
        loss_aova = out['loss_aova'] if out['loss_aova'] is not None else torch.tensor(0.0, device=DEVICE)
        if out['H_inpaint'] is not None:
            loss_inpaint = F.mse_loss(out['H_inpaint'], images)
        else:
            loss_inpaint = torch.tensor(0.0, device=DEVICE)

        loss = loss_aova + 0.5 * loss_inpaint
        loss.backward()
        optimizer.step()
        scheduler.step()

        epoch_loss += loss.item()
        pbar.set_postfix({
            'loss_total': loss.item(),
            'loss_aova': loss_aova.item(),
            'loss_inpaint': loss_inpaint.item()
        })

    avg_loss = epoch_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] avg loss: {avg_loss:.4f}")

    # Save checkpoint
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"diffugts_epoch{epoch+1}.pt")
    torch.save({
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict()
    }, ckpt_path)
    print(f"Checkpoint saved at {ckpt_path}")

print("Training finished.")
