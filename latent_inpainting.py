import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage.filters import threshold_otsu
from scipy.ndimage import binary_dilation
from typing import Optional, Tuple, List

# Utilities
def enlarge_binary_mask_np(mask_np: np.ndarray, dilation_iterations: int = 3) -> np.ndarray:
    """
    mask_np: (D, H, W) or (1, D, H, W) numpy uint8/bool
    returns enlarged mask with same spatial dims (binary np.uint8)
    """
    if mask_np.ndim == 4 and mask_np.shape[0] == 1:
        mask_np = mask_np[0]
    return binary_dilation(mask_np.astype(bool), iterations=dilation_iterations).astype(np.uint8)


def downsample_mask_to_latent(mask: torch.Tensor, latent_shape: Tuple[int, int, int]) -> torch.Tensor:
    """
    mask: torch.Tensor shape (B,1,D,H,W) or (1,D,H,W)
    latent_shape: (d, h, w)
    returns: torch.uint8 tensor (B,1,d,h,w)
    """
    if mask.dim() == 3:
        mask = mask.unsqueeze(0).unsqueeze(0).float()
    elif mask.dim() == 4:
        mask = mask.unsqueeze(1).float()
    else:
        mask = mask.float()
    d, h, w = latent_shape
    mask_ds = F.interpolate(mask, size=(d, h, w), mode="nearest")
    return mask_ds.byte()

# Feature projector
class FeatureProjector(nn.Module):
    """
    Project latent difference fr (B, C_lat, d,h,w) -> (B, text_dim, d,h,w).
    We will then dot product with text_emb (N, text_dim) to get N-class maps.
    """
    def __init__(self, in_channels: int, text_dim: int):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, text_dim, kernel_size=1)

    def forward(self, fr: torch.Tensor) -> torch.Tensor:
        # fr: (B, C_lat, d, h, w)
        return self.conv(fr)  # (B, text_dim, d, h, w)

# Main refiner class
class LatentSpaceInpaintRefiner:
    def __init__(
        self,
        autoencoder,
        diffusion_unet,
        noise_scheduler,
        controlnet: Optional[nn.Module] = None,
        device: str = "cuda",
        enlarge_iters: int = 3,
        num_infer_steps: int = 50,
        beta1: float = 0.5,
        beta2: float = 0.5,
    ):
        """
        autoencoder: object with encode(x)->z and decode(z)->image
        diffusion_unet: the trained diffusion unet (predicts noise)
        noise_scheduler: scheduler with add_noise(...) and step(...) functions (MONAI style)
        controlnet: optional controlnet for conditioning (pass to unet)
        """
        self.autoencoder = autoencoder.to(device)
        self.diffusion_unet = diffusion_unet.to(device)
        self.noise_scheduler = noise_scheduler
        self.controlnet = controlnet.to(device) if controlnet is not None else None
        self.device = device
        self.enlarge_iters = enlarge_iters
        self.num_infer_steps = num_infer_steps
        self.beta1 = beta1
        self.beta2 = beta2
        self.projector: Optional[FeatureProjector] = None

    @torch.no_grad()
    def refine(
        self,
        image: torch.Tensor,
        aova_mask: torch.Tensor,
        text_embeddings: torch.Tensor,
    ) -> dict:
        """
        Main procedure.
        Args:
          image: (B, 1, D, H, W) float tensor (range consistent with autoencoder)
          aova_mask: (B, 1, D, H, W) binary (0/1) tensor from AOVA (thresholded)
          text_embeddings: (N, text_dim) float tensor (e.g. ClinicalBERT outputs)
        Returns dict with
          'H' : pseudo-healthy image (B, C_im, D, H, W)
          'Pr': pixel residual (B,1,D,H,W)
          'Fr': feature residual maps upsampled to image space (B,N,D,H,W)
          'Rpf': combined residual (B,N,D,H,W)
          'masks_bin': binary masks per-class (B,N,D,H,W)
          'thresholds': list of per-(B,N) thresholds
        """

        assert image.dim() == 5 and aova_mask.dim() == 5, "Expect shapes (B,1,D,H,W)"
        B = image.shape[0]
        N, text_dim = text_embeddings.shape

        image = image.to(self.device)
        aova_mask = aova_mask.to(self.device).byte()
        text_embeddings = text_embeddings.to(self.device)

        # Encode to latent z
        z = self.autoencoder.encode(image)  # expect (B, C_lat, d, h, w)
        # get latent spatial dims
        _, C_lat, d, h, w = z.shape

        # lazy init projector if needed
        if self.projector is None:
            self.projector = FeatureProjector(in_channels=C_lat, text_dim=text_dim).to(self.device)

        # enlarge masks (use numpy dilation per-sample)
        mask_ds_list = []
        for i in range(B):
            mask_np = aova_mask[i, 0].cpu().numpy()
            mask_enlarged_np = enlarge_binary_mask_np(mask_np, dilation_iterations=self.enlarge_iters)
            mask_enlarged_t = torch.from_numpy(mask_enlarged_np).unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)
            mask_latent = downsample_mask_to_latent(mask_enlarged_t, (d, h, w)).to(self.device)  # (1,1,d,h,w)
            mask_ds_list.append(mask_latent)
        mask_latent = torch.cat(mask_ds_list, dim=0)  # (B,1,d,h,w)

        # latent inpainting sampling loop (training-free)
        z0 = self._latent_inpaint(z=z, mask_latent=mask_latent, steps=self.num_infer_steps)

        # decode pseudo-healthy image
        H = self.autoencoder.decode(z0)  # (B, C_im, D, H, W)

        # pixel residual (I - H), normalize both similarly before subtraction
        I_norm = (image - image.mean(dim=[2,3,4], keepdim=True)) / (image.std(dim=[2,3,4], keepdim=True) + 1e-6)
        H_norm = (H - H.mean(dim=[2,3,4], keepdim=True)) / (H.std(dim=[2,3,4], keepdim=True) + 1e-6)
        Pr = torch.abs(I_norm - H_norm)  # (B, C_im, D, H, W)
        # if multi-channel image, reduce to single channel by mean for residual fusion
        Pr_single = Pr.mean(dim=1, keepdim=True)  # (B,1,D,H,W)

        # feature residual fr = z - z0 (B, C_lat, d, h, w)
        fr = z - z0

        # project fr -> per-text-dim maps: (B, text_dim, d, h, w)
        fr_proj = self.projector(fr)  # (B, text_dim, d, h, w)

        # compute Fr per text prompt by dot-product: Fr_i = sum_c fr_proj * text_emb[i,c]
        # result: (B, N, d, h, w)
        # normalize text embeddings
        txt_norm = F.normalize(text_embeddings, dim=-1)  # (N, text_dim)
        fr_proj_norm = F.normalize(fr_proj, dim=1)  # normalize channel-wise to stabilize dot
        # compute per-class maps
        Fr_latent = torch.einsum("bcdhw,nc->bndhw", fr_proj_norm, txt_norm)  # (B,N,d,h,w)

        # upsample Fr_latent to image resolution (D,H,W)
        Fr_up = F.interpolate(Fr_latent.unsqueeze(1), size=image.shape[2:], mode="trilinear", align_corners=False)
        # now Fr_up shape: (B,1,N,D,H,W) after unsqueeze; reorder to (B,N,D,H,W)
        Fr_up = Fr_up.squeeze(1)  # (B,N,D,H,W)

        # combine residuals per class: Rpf = beta1 * Pr_single + beta2 * Fr_up
        # Broadcast Pr_single (B,1,D,H,W) to (B,N,D,H,W)
        Pr_bcast = Pr_single.expand(-1, N, -1, -1, -1)
        Rpf = self.beta1 * Pr_bcast + self.beta2 * Fr_up

        # convert to binary mask via Otsu per-sample, per-class
        masks_bin = torch.zeros_like(Rpf, dtype=torch.uint8)
        thresholds = torch.zeros((B, N), dtype=torch.float32)
        Rpf_cpu = Rpf.detach().cpu().numpy()
        for i in range(B):
            for j in range(N):
                arr = Rpf_cpu[i, j]  # (D,H,W)
                try:
                    t = threshold_otsu(arr.flatten())
                except Exception:
                    t = float(np.mean(arr))
                maskb = (arr >= t).astype(np.uint8)
                masks_bin[i, j] = torch.from_numpy(maskb)
                thresholds[i, j] = float(t)

        # return everything useful
        return {
            "H": H,                        # (B,C_im,D,H,W)
            "Pr": Pr_single,               # (B,1,D,H,W)
            "Fr": Fr_up,                   # (B,N,D,H,W)
            "Rpf": Rpf,                    # (B,N,D,H,W)
            "masks_bin": masks_bin,        # (B,N,D,H,W) uint8
            "thresholds": thresholds       # (B,N)
        }

    @torch.no_grad()
    def _latent_inpaint(self, z: torch.Tensor, mask_latent: torch.Tensor, steps: int = 50) -> torch.Tensor:
        """
        Training-free latent inpainting following Eq (3-5) in the paper.
        z: (B, C_lat, d, h, w)
        mask_latent: (B,1,d,h,w) binary (1=inpaint/tumor region)
        returns z0: (B, C_lat, d, h, w)
        """
        B = z.shape[0]
        device = z.device

        # initialize z_t by adding noise at final timestep
        max_t = steps - 1
        noise = torch.randn_like(z)
        t_init = torch.full((B,), max_t, device=device, dtype=torch.long)

        zt = self.noise_scheduler.add_noise(original_samples=z, noise=noise, timesteps=t_init)

        # reverse loop (we go from t=max_t down to 0)
        for t in reversed(range(0, steps)):
            t_tensor = torch.full((B,), t, device=device, dtype=torch.long)

            # predict noise with unet; include controlnet if present (controlnet API might differ)
            if self.controlnet is not None:
                control = self.controlnet(zt, t_tensor)
                model_pred = self.diffusion_unet(zt, t_tensor, control=control)
            else:
                model_pred = self.diffusion_unet(zt, t_tensor)

            # step: returns object with prev_sample as mean estimate (MONAI style)
            step_out = self.noise_scheduler.step(model_output=model_pred, timestep=t_tensor, sample=zt)
            z_tumor = step_out.prev_sample  # model-driven inpaint for masked region

            # z_other sampled from distribution around original z at this timestep
            # use add_noise to get noisy version of original z at timestep t
            noise_other = torch.randn_like(z)
            z_other = self.noise_scheduler.add_noise(original_samples=z, noise=noise_other, timesteps=t_tensor)

            # mix according to mask_latent
            mask_b = mask_latent  # (B,1,d,h,w)
            zt = mask_b * z_tumor + (1 - mask_b) * z_other

        # after loop zt is z0 latent
        return zt
