import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

# Dice Loss
class DiceLoss(nn.Module):
    def __init__(self, sigmoid: bool = True):
        super().__init__()
        self.sigmoid = sigmoid

    def forward(self, pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6):
        if self.sigmoid:
            pred = torch.sigmoid(pred)

        pred_flat = pred.reshape(pred.shape[0], -1)
        target_flat = target.reshape(target.shape[0], -1)

        intersection = (pred_flat * target_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)

        dice = (2.0 * intersection + smooth) / (union + smooth)
        return 1.0 - dice.mean()

# Text Reducer (768 -> per-level channel sizes)
class TextDimensionReducer(nn.Module):
    def __init__(self, input_dim: int = 768, output_dims: List[int] = [64, 128, 256]):
        super().__init__()
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.GELU(),
                nn.Linear(input_dim // 2, out_dim),
                nn.LayerNorm(out_dim)
            ) for out_dim in output_dims
        ])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        x: (N, text_dim)
        returns: list of length L where element l is (N, out_dim_l)
        """
        return [mlp(x) for mlp in self.mlps]

# AOVA
class AOVAModuleWithLosses(nn.Module):
    def __init__(self,
                 feature_channels_list: List[int] = [64, 128, 256],
                 text_dim: int = 768,
                 num_categories: int = 2,
                 temperature: float = 0.07,
                 target_D: int = 128,
                 target_H: int = 512,
                 target_W: int = 512):
        """
        feature_channels_list: channel dims for each image level (C1, C2, C3)
        text_dim: dimension of CLIP/text embedding (and class embedding dim)
        num_categories: N (e.g., 2)
        temperature: τ for contrastive/scaling
        target_D/H/W: canonical AOVA map resolution to which all levels are resized
        """
        super().__init__()

        assert len(feature_channels_list) >= 1

        self.feature_channels_list = feature_channels_list
        self.text_dim = text_dim
        self.num_categories = num_categories
        self.temperature = temperature

        self.target_D = target_D
        self.target_H = target_H
        self.target_W = target_W

        # Reduce CLIP text embedding to per-level key dims
        self.text_reducer = TextDimensionReducer(input_dim=text_dim, output_dims=feature_channels_list)

        # Q and K projections at each level (operate on channel dimension)
        self.W_q_proj = nn.ModuleList([nn.Linear(c, c) for c in feature_channels_list])
        self.W_k_proj = nn.ModuleList([nn.Linear(c, c) for c in feature_channels_list])

        # class embedding MLP: input is pooled D (target_D) -> output text_dim
        self.class_embedding_mlp = nn.Sequential(
            nn.Linear(target_D, max(target_D // 2, 1)),
            nn.GELU(),
            nn.Linear(max(target_D // 2, 1), text_dim),
            nn.LayerNorm(text_dim)
        )

        # anomaly classifier: Rd -> scalar probability
        self.anomaly_classifier = nn.Sequential(
            nn.Linear(text_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        # losses
        self.bce = nn.BCELoss()
        self.dice = DiceLoss(sigmoid=True)

    # Cross-attention per level -> produce attention maps per level
    # each output: (B, N, D_l, H_l, W_l)
    def _compute_cross_attention_per_level(self,
                                           features: List[torch.Tensor],
                                           text_reduced: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        features: list of F_l: (B, C_l, D_l, H_l, W_l)
        text_reduced: list of txt_l: (N, C_l)
        returns: list of att_l resized to their native (D_l, H_l, W_l) -> (B, N, D_l, H_l, W_l)
        """
        att_maps = []
        for F_l, Wq, Wk, txt_l in zip(features, self.W_q_proj, self.W_k_proj, text_reduced):
            B, C_l, D_l, H_l, W_l = F_l.shape
            N = txt_l.shape[0]

            # Q: prepare image queries -> apply linear on the channel dimension
            Q = F_l.permute(0, 2, 3, 4, 1).contiguous()   # (B, D_l, H_l, W_l, C_l)
            Q = Wq(Q)  # applied on last dim -> (B, D_l, H_l, W_l, C_l)

            # K: text reduced projection (N, C_l) -> project to same C_l
            K = Wk(txt_l)  # (N, C_l)

            # Attention logits: (B, D_l, H_l, W_l, N)
            att_logits = torch.einsum("bdhwc,nc->bdhwn", Q, K) / (C_l ** 0.5)

            att = F.softmax(att_logits, dim=-1)  # softmax over N
            # reorder to (B, N, D_l, H_l, W_l)
            att = att.permute(0, 4, 1, 2, 3).contiguous()
            att_maps.append(att)

        return att_maps

    # input att_list: list of (B, N, D_l, H_l, W_l)
    # output: (B, N, target_D, target_H, target_W)
    def build_aova_maps(self, att_list):
        """
        att_list: list of 3 tensors
            each (B, N, D_l, H_l, W_l)

        returns:
            att: (B, N, 128, 128, 128)
        """

        target_D = self.target_D
        target_H = self.target_H
        target_W = self.target_W

        resized_maps = []

        for att in att_list:
            B, N, D, H, W = att.shape

            # 1) Resize spatial dims (H, W) → (128,128)
            x = att.reshape(B * N, D, H, W)     # → (BN, D, H, W)

            x = F.interpolate(
                x,
                size=(target_H, target_W),      # only H,W
                mode="bilinear",
                align_corners=False
            )
            # (BN, D, 128, 128)

            x = x.reshape(B, N, D, target_H, target_W)

            # -----------------------------------------
            # 2) Resize channel depth D → 128 using 1D interpolation
            # -----------------------------------------
            # bring D to last dim
            x = x.permute(0,1,3,4,2)  # (B,N,128,128,D)

            x = x.reshape(B*N*target_H*target_W, D).unsqueeze(1)  # (BNHW,1,D)

            x = F.interpolate(
                x,
                size=(target_D,),            # only D
                mode="linear",
                align_corners=False
            )

            # reshape back
            x = x.squeeze(1).reshape(B, N, target_H, target_W, target_D)

            # permute back to (B,N,D,H,W)
            x = x.permute(0,1,4,2,3)

            resized_maps.append(x)

        att = resized_maps[0] + resized_maps[1] + resized_maps[2]

        return att   # (B, N, 128,128,128)


    # Convert AOVA maps -> class embeddings gi (B, N, text_dim)
    def _get_class_embeddings(self, aova_maps: torch.Tensor) -> torch.Tensor:
        # aova_maps: (B, N, target_D, target_H, target_W)
        B, N, D, H, W = aova_maps.shape

        # Adaptive average pool to (D,1,1) -> then squeeze to (B, N, D)
        pooled = F.adaptive_avg_pool3d(aova_maps, (D, 1, 1)).view(B, N, D)

        # MLP expects last dim = target_D, Linear will operate on that last dim
        # class_embedding_mlp applies to (..., target_D) and returns (..., text_dim)
        gi = self.class_embedding_mlp(pooled)  # (B, N, text_dim)
        return gi

    # Anomaly score: sigmoid( MLP( max_over_classes(g) ) )
    def _compute_anomaly_scores(self, gi: torch.Tensor) -> torch.Tensor:
        g_max = gi.max(dim=1).values  # (B, text_dim)
        score = self.anomaly_classifier(g_max)  # (B, 1) after Sigmoid
        return score

    # CLIP-style symmetric contrastive loss
    # gi: (B, N, d)
    # text_emb: (N, d)
    # labels: (B,) values in {0..N-1}
    def _contrastive_loss(self, gi: torch.Tensor, text_emb: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        B, N, d = gi.shape
        device = gi.device

        # Normalize
        gi_norm = F.normalize(gi, dim=-1)            # (B, N, d)
        text_norm = F.normalize(text_emb, dim=-1)    # (N, d)

        loss_total = 0.0
        for i in range(B):
            sim = (gi_norm[i] @ text_norm.T) / self.temperature  # (N, N)
            targets = torch.arange(N, device=device)
            loss_img = F.cross_entropy(sim, targets)
            loss_txt = F.cross_entropy(sim.T, targets)
            loss_total += 0.5 * (loss_img + loss_txt)

        return loss_total / float(B)

    def _dice_loss(self, masks: torch.Tensor, gt: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = masks.device
        B = masks.shape[0]
        total = 0.0
        count = 0

        # Normalize gt shape to (B, N, D, H, W)
        if gt.dim() == 4:  # (B, D, H, W) or (B, 1, D, H, W) etc
            gt = gt.unsqueeze(1)

        for i in range(B):
            lbl = int(labels[i].item())
            if lbl != 0 and gt[i].sum() > 0:
                if gt.shape[1] == masks.shape[1]:
                    gt_seg = gt[i, lbl:lbl+1]  # (1, D, H, W)
                else:
                    gt_seg = gt[i:i+1]  # (1, D, H, W)

                pred_seg = masks[i, lbl:lbl+1]  # (1, D, H, W)
                loss = self.dice(pred_seg.unsqueeze(0), gt_seg.unsqueeze(0))
                total += loss
                count += 1

        if count == 0:
            return torch.tensor(0.0, device=device)
        return total / float(count)

    # Full loss wrapper
    def compute_losses(self, out: dict, text_embeddings: torch.Tensor,
                       labels: torch.Tensor, seg_masks: Optional[torch.Tensor] = None) -> dict:
        """
        out: dict with keys:
            'aova_maps' : (B, N, D, H, W)
            'class_embeddings' : (B, N, d)
            'anomaly_scores' : (B, 1)
            'binary_masks' : (B, N, D, H, W)
        labels: (B,)  0..N-1
        """
        device = text_embeddings.device
        labels = labels.to(device)

        loss_ano = self.bce(out["anomaly_scores"].squeeze(1), labels.float().to(device))

        loss_sim = self._contrastive_loss(out["class_embeddings"], text_embeddings.to(device), labels)

        if seg_masks is not None:
            loss_dice = self._dice_loss(out["binary_masks"], seg_masks.to(device), labels)
        else:
            loss_dice = torch.tensor(0.0, device=device)

        loss_total = loss_ano + loss_sim + loss_dice

        return {
            "loss_ano": loss_ano,
            "loss_sim": loss_sim,
            "loss_dice": loss_dice,
            "loss_aova": loss_total
        }

    def forward(self,
                image_features: List[torch.Tensor],
                text_embeddings: torch.Tensor,
                img_labels: Optional[torch.Tensor] = None,
                seg_labels: Optional[torch.Tensor] = None) -> dict:
        """
        image_features: list of L tensors [(B, C1, D1, H1, W1), ...]
        text_embeddings: (N, text_dim)
        img_labels: (B,) values 0..N-1 (image-level)
        seg_labels: optional (B, N, D, H, W) or (B, 1, D, H, W)
        """
        B = image_features[0].shape[0]
        N = text_embeddings.shape[0]
        assert N == self.num_categories, "text_emb count mismatch"

        # reduce text to per-level keys
        text_reduced = self.text_reducer(text_embeddings)  # list of (N, C_l)

        # compute cross-attention per level
        att_per_level = self._compute_cross_attention_per_level(image_features, text_reduced)
        # att_per_level: list of (B, N, D_l, H_l, W_l)

        # build AOVA maps by resizing + summing
        aova_maps = self.build_aova_maps(att_per_level)  # (B, N, target_D, target_H, target_W)

        # class embeddings
        class_embeddings = self._get_class_embeddings(aova_maps)  # (B, N, d)

        # anomaly scores
        anomaly_scores = self._compute_anomaly_scores(class_embeddings)  # (B, 1)

        # binary masks
        binary_masks = torch.sigmoid(aova_maps)

        out = {
            "aova_maps": aova_maps,
            "class_embeddings": class_embeddings,
            "anomaly_scores": anomaly_scores,
            "binary_masks": binary_masks
        }

        if img_labels is not None:
            losses = self.compute_losses(out, text_embeddings, img_labels, seg_labels)
            out.update(losses)

        return out


