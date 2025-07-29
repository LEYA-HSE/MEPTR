# Multimodal Cross-Domain Model for MEPTR (Fusion Branch)

This branch implements the **Multimodal Cross-Domain Model** described in our paper published in **Expert Systems with Applications (ESWA), 2025**:

> [Elena Ryumina](https://scholar.google.com/citations?user=DOBkQssAAAAJ), [Alexandr Axyonov](https://scholar.google.com/citations?user=Hs95wd4AAAAJ), Darya Koryakovskaya, Timur Abdulkadirov, Angelina Egorova, Sergey Fedchin, Alexander Zaburdaev, [Dmitry Ryumin](https://scholar.google.com/citations?user=LrTIp5IAAAAJ)  
> HSE University

---

## 📌 Overview

The Multimodal Cross-Domain Model integrates predictions and feature embeddings from five modality-specific Unimodal Cross-Domain Models (Face, Body, Scene, Audio, and Text). It leverages advanced multimodal fusion techniques designed specifically for Multitask Emotion and Personality Traits Recognition (MEPTR).

---

## 🔧 Model Components

This branch contains the following model components:

- **Graph Attention Fusion**  
  Combines embeddings from multiple modalities, modeling inter-modality relationships through learnable graph-based attention mechanisms.

- **Task-Specific Query-Based Multi-Head Cross-Attention Fusion**  
  Performs task-specific cross-attention, selectively aggregating relevant information across modalities separately for emotion recognition (classification) and personality traits assessment (regression).

- **Task-Specific Predict Projectors**  
  Project averaged predictions from unimodal models into a shared latent space, facilitating joint multimodal decision-making.

- **Task-Specific Guide Banks**  
  Sets of learned embeddings representing each class (emotion labels and personality traits). These embeddings help align modality-specific embeddings through cosine similarity.

- **Multitask Joint Optimization**  
  Simultaneously optimizes multitask predictions using a combined cross-entropy loss (for emotion classification) and mean absolute error loss (for personality traits regression).

---

## 🌿 Input Modality Requirements

To train or evaluate the multimodal model, provide features and predictions from each **Unimodal Cross-Domain Model** (trained independently in separate branches):

- **Face modality**: CLIP embeddings + Mamba temporal encoder (from `face_trainer`)
- **Body modality**: CLIP embeddings + Mamba temporal encoder (from `body_trainer`)
- **Scene modality**: CLIP embeddings + Transformer temporal encoder (from `scene_trainer`)
- **Audio modality**: Wav2Vec2 embeddings + Mamba temporal encoder (from `audio_trainer`)
- **Text modality**: BGE-en embeddings + Transformer temporal encoder (from `text_trainer`)
