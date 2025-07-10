# coding: utf-8
"""
train.py  (multitask supra-modal training — multimodal loader variant)
------------------------------------------------------------------
*   Uses a single multimodal DataLoader yielding batches with:
      batch['features']: dict modality->Tensor([B,D])
      batch['labels']:   dict with 'emotion' (one-hot) and/or 'personality' (scores)
*   Two-stage pipeline per epoch:
      1) Alignment phase (iterate modalities, then over loader)
      2) Concept-guided phase (borrow strength via guidance set)
*   Heads:
      - Emotion classification: 7-way CrossEntropy
      - Personality regression: 5-dim MSE

Call from main.py:

    from train import train as supra_train
    supra_train(cfg, train_loader)

"""
from __future__ import annotations
import random
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.supra_models import SupraMultitaskModel
from supra_train_utils import (
    alignment_train_step,
    build_guidance_set,
    concept_guided_train_step,
)

# ────────────────────────────────────────────────────────────── #

def seed_everything(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ────────────────────────────────────────────────────────────── #

def train(cfg, mm_loader: DataLoader):
    """
    Args:
        cfg: ConfigLoader instance (must have device, batch_size, epochs, lr, lambda_w, checkpoint_dir)
        mm_loader: DataLoader yielding dicts with 'features' and 'labels'
    """
    seed_everything(cfg.random_seed)

    device = cfg.device
    # peek first batch to get modalities and dims
    first = next(iter(mm_loader))
    modalities = list(first['features'].keys())
    input_dims = {m: first['features'][m].shape[1] for m in modalities}

    # build model + optimizer
    model = SupraMultitaskModel(
        input_dims=input_dims,
        hidden_dim=cfg.hidden_dim,
        emo_out_dim=7,
        pkl_out_dim=5,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-5)

    # two-stage training
    for epoch in range(cfg.epochs):
        print(f"Epoch {epoch+1}/{cfg.epochs}")
        model.train()

        # 1) Alignment phase
        for mod in modalities:
            total_loss = 0.0
            for batch in tqdm(mm_loader, desc=f"Align {mod}"):
                feat = batch['features'][mod].to(device)
                labels = {k: v.to(device) for k, v in batch['labels'].items()}
                mini_batch = {'features': {mod: feat}, 'labels': labels, 'modality': mod}
                total_loss += alignment_train_step(model, optimizer, mini_batch,
                                                 beta1=cfg.beta1, beta2=cfg.beta2)
            print(f"  {mod}: align_loss={total_loss/len(mm_loader):.4f}")

        # build guidance set (emotion & personality features)
        # reuse same loader for all modalities
        loaders_dict = {m: mm_loader for m in modalities}
        guidance = build_guidance_set(model, loaders_dict,
                                      top_k=cfg.top_k, device=device)

        # 2) Concept-guided phase
        for mod in modalities:
            total_loss = 0.0
            for batch in tqdm(mm_loader, desc=f"Concept {mod}"):
                feat = batch['features'][mod].to(device)
                labels = {k: v.to(device) for k, v in batch['labels'].items()}
                mini_batch = {'features': {mod: feat}, 'labels': labels, 'modality': mod}
                total_loss += concept_guided_train_step(model, optimizer,
                                                       mini_batch, guidance,
                                                       lambda_weight=cfg.lambda_w,
                                                       device=device)
            print(f"  {mod}: concept_loss={total_loss/len(mm_loader):.4f}")

    # save final model
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    ckpt_path = Path(cfg.checkpoint_dir)/"supra_multitask_final.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"\n✔ Model saved to {ckpt_path}")
