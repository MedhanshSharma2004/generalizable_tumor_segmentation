import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import torchvision
from torchvision.models.feature_extraction import create_feature_extractor
from aova_map import AOVAModuleWithLosses
from vae_adapter import Feature_Adapter
from latent_inpainting import LatentSpaceInpaintRefiner

class DiffuGTS(nn.Module):
    """
    DiffuGTS model using:
    - VAE feature extraction (with intermediate layers)
    - Feature adaptation (residual)
    - AOVA map construction using text embeddings
    - Latent-space inpainting refinement
    """
    def __init__(self, autoencoder, latent_inpaint_refiner: LatentSpaceInpaintRefiner, alpha=0.1, device='cuda'):
        super().__init__()
        self.device = device

        # Feature Adapters
        self.feature_adapters = nn.ModuleList([
            Feature_Adapter(64, alpha=alpha),
            Feature_Adapter(128, alpha=alpha),
            Feature_Adapter(256, alpha=alpha)
        ])

        # AOVA Maps
        self.aova = AOVAModuleWithLosses().to(device)

        # VAE
        self.autoencoder = autoencoder
        self.vae_encoder = autoencoder.encoder
        self.vae_decoder = autoencoder.decoder
        self.return_nodes = {
            'blocks.3': 'broad_features_64',
            'blocks.6': 'mid_features_128',
            'blocks.8': 'low_features_256',
            'blocks.10': 'latent_z'
        }

        # Latent-space inpainting refiner
        self.refiner = latent_inpaint_refiner

    def forward(self, x, text_embeddings, class_labels=None, spacing=None):
        """
        x: input 3D volume (B, C, D, H, W)
        text_embeddings: tensor of shape (N_text, text_dim)
        class_labels: optional image-level class labels
        spacing: optional conditioning
        """
        # Encode input using frozen VAE
        feature_extractor = create_feature_extractor(self.vae_encoder, self.return_nodes)
        features = feature_extractor(x)
        feat_l3, feat_l6, feat_l8, z = features['blocks.3'], features['blocks.6'], features['blocks.8'], features['blocks.10']

        # Apply feature adapters (residual learning)
        feat_l3 = self.feature_adapters[0](feat_l3)
        feat_l6 = self.feature_adapters[1](feat_l6)
        feat_l8 = self.feature_adapters[2](feat_l8)

        # Generate AOVA maps
        aova_out = self.aova(
            image_features=[feat_l3, feat_l6, feat_l8],
            text_embeddings=text_embeddings,
            img_labels=class_labels
        )

        binary_masks = aova_out['binary_masks']  # (B, N, D, H, W)

        # Latent-space inpainting refinement
        if binary_masks is not None:
            inpaint_out = self.refiner.refine(
                image=x, 
                aova_mask=binary_masks, 
                text_embeddings=text_embeddings
            )
        else:
            inpaint_out = {
                "H": None,
                "Pr": None,
                "Fr": None,
                "Rpf": None,
                "masks_bin": None,
                "thresholds": None
            }

        # Decode latent to reconstruct input (optional)
        recon = self.vae_decoder(z)

        return {
            # AOVA outputs
            'recon': recon,
            'binary_masks': binary_masks,
            'loss_aova': aova_out.get('loss_aova', None),
            'loss_dice': aova_out.get('loss_dice', None),
            'loss_sim': aova_out.get('loss_sim', None),
            'loss_ano': aova_out.get('loss_ano', None),

            # Latent-space inpainting outputs
            'H_inpaint': inpaint_out["H"],
            'Pr': inpaint_out["Pr"],
            'Fr': inpaint_out["Fr"],
            'Rpf': inpaint_out["Rpf"],
            'masks_inpaint_bin': inpaint_out["masks_bin"]
        }

