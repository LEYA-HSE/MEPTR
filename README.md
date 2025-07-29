# `fusion` Branch

This branch contains the **final multimodal fusion pipeline** for multitask emotion and personality trait recognition, as proposed in the ESWA 2025 paper.

## 🧠 Overview

The fusion architecture integrates pretrained unimodal encoders (text, audio, face, body, and scene) into a unified multitask prediction framework using:

- **Query-Guided Cross-Modal Decoding**: A shared transformer-based decoder receives task-specific query embeddings and attends over modality-specific encoded representations. The decoder performs inter-modal attention without requiring explicit temporal synchronization.
  
- **Guide Banks**: For each modality, we construct learnable latent memory banks that store class-aware modality-level priors. These banks guide the attention mechanism during both training and inference, enabling more robust generalization in low-resource settings.

- **Task-Specific Heads**: Fused representations are passed through separate classification heads for:
  - **Emotion recognition** (multi-label classification)
  - **Personality trait regression** (continuous trait prediction)

- **Multitask Optimization**: A joint training setup using:
  - Weighted Focal Loss and mWACC for emotion tasks
  - CCC loss and MAE for personality regression

## 🧪 Experimental Setup

- Fusion uses frozen or partially fine-tuned encoders from:
  - `text_trainer`: Jina + Mamba
  - `audio_trainer`: wav2vec2 + Mamba
  - `face_trainer`: CNN + Transformer
  - `body_trainer`: Pose-RNN
  - `scene_trainer`: Contextual CNN features

- The model was evaluated on:
  - **CMU-MOSEI** for emotion classification
  - **ChaLearn First Impressions v2** for personality trait regression

- All reported results (Tables 4–7) in the publication are derived from experiments in this branch.

## ⚡ Inference and Real-Time Compatibility

The fusion model achieves **real-time inference** with **RTA < 1**, enabling deployment in interactive multimodal systems.

---

> For reproducibility, all hyperparameters, checkpoint paths, and evaluation metrics are included in the branch.  
> Please cite the original publication if using this branch in academic work.
