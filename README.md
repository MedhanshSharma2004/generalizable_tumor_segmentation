# DiffuGTS: Generalizable Tumor Segmentation with Anomaly-Aware Open-Vocabulary Attention Maps


This repository contains an implementation of **DiffuGTS**, a framework for **zero-shot, generalizable tumor segmentation** across multiple anatomical regions and imaging modalities, based on the paper:

**Advancing Generalizable Tumor Segmentation with Anomaly-Aware Open-Vocabulary Attention Maps and Frozen Foundation Diffusion Models**  
*Yankai Jiang, Peng Zhang, Donglin Yang, Yuan Tian, Hai Lin, Xiaosong Wang*


---

## Overview

DiffuGTS addresses the challenge of generalizable tumor segmentation (GTS), where the goal is to train a **single model capable of zero-shot segmentation** across diverse tumor types and imaging modalities without relying on fully annotated datasets.

Key innovations include:

- **Anomaly-Aware Open-Vocabulary Attention (AOVA) Maps**  
  - Uses frozen medical foundation diffusion models (MFDMs) to extract rich anatomical features.  
  - Leverages text prompts describing normal and abnormal organs to generate attention maps for zero-shot tumor localization.  

- **Mask Refinement with Frozen Diffusion Models**  
  - Synthesizes **pseudo-healthy images** from pathological scans via latent-space inpainting.  
  - Applies **pixel-level and feature-level residual learning** to refine segmentation masks, improving accuracy and generalization.

- **Zero-Shot Generalization**  
  - Capable of segmenting unseen tumors across multiple anatomical regions (brain, lung, liver, kidney, pancreas, colon) and imaging modalities (CT, MRI).

---

## Architecture

The DiffuGTS pipeline:

1. **Input**: 3D medical image + text prompts describing normal and abnormal organs.  
2. **Feature Extraction**: Multi-scale features from the frozen VAE encoder of a medical diffusion model.  
3. **AOVA Map Generation**: Cross-modal attention between image features and text embeddings to detect anomalies.  
4. **Pseudo-Healthy Synthesis**: Latent-space inpainting to generate a healthy counterpart.  
5. **Residual Learning**: Combines pixel-level and feature-level residuals to refine segmentation masks.  
6. **Output**: High-quality tumor segmentation mask.

---

git clone https://github.com/MedhanshSharma2004/generalizable_tumor_segmentation.git
cd generalizable_tumor_segmentation
