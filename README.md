# A Semi-Supervised Multimodal Framework for Multitask Emotion and Personality Traits Recognition Using Cross-Domain Learning

This repository accompanies the publication in **Expert Systems with Applications (ESWA), 2025**:

> [Elena Ryumina](https://scholar.google.com/citations?user=DOBkQssAAAAJ), [Alexandr Axyonov](https://scholar.google.com/citations?user=Hs95wd4AAAAJ), Timur Abdulkadirov, Darya Koryakovskaya, Svetlana Gorovaya, Anna Bykova, Dmitry Vikhorev, [Dmitry Ryumin](https://scholar.google.com/citations?user=LrTIp5IAAAAJ)
> 
> HSE University

---

## 🧠 Abstract

The growing demand for intelligent human-computer interaction systems has driven the need for personalized solutions. Early research addressed this issue through Emotion Recognition (ER) methods, while current approaches focus on assessing individual Personality Traits (PTs). However, effective systems must integrate both capabilities, requiring large-scale corpora annotated for both tasks, which are currently unavailable. We introduce a semi-supervised multimodal framework for Multitask Emotion and Personality Traits Recognition (MEPTR) with a three-stage learning strategy, combining unimodal single-domain, unimodal cross-domain, and multimodal cross-domain models. We further enhance multimodal fusion by proposing Graph Attention Fusion and Task-Specific Query-Based Multi-Head Cross-Attention Fusion, as well as task-specific Predict Projectors and Guide Banks. This enables the model to effectively integrate heterogeneous and semi-labeled data. The framework is evaluated on two on two large-scale corpora, CMU Multimodal Opinion Sentiment and Emotion Intensity (MOSEI) for emotion recognition and ChaLearn First Impressions v2 corpora (FIv2) for Personality Traits (PTs) assessment, showcasing its potential application in personalized human-computer interaction systems. In single-domain learning, our model achieves mean Average Weighted Accuracy (mWACC) of 70.26 on MOSEI and mean Accuracy (mACC) of 92.88 on FIv2, outperforming state-of-the-art results. In contrast, cross-domain learning demonstrates reduced performance, yielding mWACC of 64.26 on MOSEI and mACC of 92.00 on FIv2. Our results highlight the complexity of both single-domain learning (the problem of overfitting on single-domain) and cross-domain learning (the problem of adapting a model to different domains varying modality informativeness). We also study the relationship between these phenomena, finding that negative emotions, especially Sadness, are negatively correlated with high-level PTs scores, whereas Happiness is positively associated with these levels.

---

## ✨ Highlights

- 📦 **Multitask learning** for simultaneous emotion and personality trait recognition  
- 🌐 **Cross-domain framework** handling heterogeneous datasets  
- 🧊 **Modular unimodal encoders** with Mamba/Transformer/GAT backbones  
- 🔗 **Query-based multimodal fusion** with cross-attention and guide banks  
- 🧪 **Superior performance** on CMU-MOSEI and ChaLearn v2 under low-label supervision  
- ⚡ **Real-time inference** (RTA < 1) enabling practical applications  

---

## 🌳 Branch Descriptions

| Branch | Description |
|--------|-------------|
| `main` | Default branch with descriptions for the ESWA 2025 publication. |
| `audio_trainer` | Audio modality trainer with Mamba/Transformer-based encoders. |
| `text_trainer` | Text-only classification pipeline using Jina embeddings and Mamba classifier. |
| `face_trainer` | Facial expression recognition component, used in fusion. |
| `body_trainer` | Body posture and movement feature extraction for personality traits. |
| `scene_trainer` | Scene-aware embeddings via CNN-based encoders for environment context. |
| `fusion` | Cross-modal fusion models integrating outputs from all modalities. Includes query-heads, graph fusion, and joint optimization pipelines. |

---

## 🏋️‍♂️ Training

The training process consists of **three phases**, implemented across separate branches:

### 1. **Unimodal Pretraining**
Each modality is trained independently:
- [`text_trainer`](../tree/text_trainer): text classification with Jina/Mamba.
- [`audio_trainer`](../tree/audio_trainer): audio classification with wav2vec2 + Mamba.
- [`face_trainer`](../tree/face_trainer): facial features via CNN + Transformer.
- [`body_trainer`](../tree/body_trainer): body motion encodings via keypoints and RNNs.
- [`scene_trainer`](../tree/scene_trainer): scene embeddings using pre-trained CNNs.

### 2. **Cross-Domain Adaptation**
Model checkpoints are transferred between datasets (e.g., emotion → personality), aligning shared latent representations with **cross-domain losses**.

### 3. **Multimodal Fusion**
Branch [`fusion`](../tree/fusion) performs:
- Feature projection and alignment,
- Guide Bank construction per task,
- Query-driven attention-based fusion.

---

## 📝 Citation

If you use this work, please cite the following:

```bibtex
@article{ryumina2025semisupervised,
  title   = {A Semi-Supervised Multimodal Framework for Multitask Emotion and Personality Traits Recognition Using Cross-Domain Learning},
  author  = {Ryumina, Elena and Axyonov, Alexandr and Koryakovskaya, Darya and Abdulkadirov, Timur and Egorova, Angelina and Fedchin, Sergey and Zaburdaev, Alexander and Ryumin, Dmitry},
  journal = {Expert Systems with Applications},
  year    = {2025},
  note    = {Under review}
}
