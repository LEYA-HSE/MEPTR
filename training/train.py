# coding: utf-8
"""
train.py  – Supramodal multitask training (multimodal-loader вариант)
---------------------------------------------------------------------
Пайплайн на один «сквозной» DataLoader:
    1) Alignment-фаза – учим общий эмо-энкодер + ортогональность.
    2) Concept-guided-фаза – «подтягиваем» PKL через guidance-set.

Ожидаемый batch:
    batch = {
        'features': {mod: Tensor|None, ...},  # [B,D] или None
        'labels':   {'emotion': Tensor|None,
                     'personality': Tensor|None}
    }

Использование (в main.py):
    from train import train as supra_train
    supra_train(cfg, train_loader)
"""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.supra_models import SupraMultitaskModel
from .supra_train_utils import (
    alignment_train_step,
    build_guidance_set,
    concept_guided_train_step,
)


# ─────────────────────────────── utils ────────────────────────────────
def seed_everything(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def first_non_none_feature(batch, modality):
    """
    Возвращает Tensor([B,D]) для `modality`, если в батче он не None,
    иначе – None.
    """
    feat = batch["features"].get(modality)
    return feat if feat is not None else None


def move_labels_to_device(lbls: Dict[str, torch.Tensor | None],
                          device: torch.device):
    """
    Переносим только те метки, что не None.
    """
    return {k: (v.to(device) if v is not None else None)
            for k, v in lbls.items()}


# ────────────────────────── основной train() ──────────────────────────
def train(cfg, mm_loader: DataLoader):
    """
    Args
    ----
    cfg        – объект-конфиг (attrs: device, random_seed, epochs,
                 hidden_dim, lr, beta1, beta2, lambda_w, top_k,
                 checkpoint_dir)
    mm_loader  – единый DataLoader, отдающий multimodal-батчи
    """

    seed_everything(cfg.random_seed)
    device = cfg.device

    # ── 0. Определяем живые модальности и их размерности ──────────────
    input_dims: Dict[str, int] = {}
    for batch in mm_loader:
        for mod, feat in batch["features"].items():
            if feat is not None and mod not in input_dims:
                input_dims[mod] = feat.shape[1]
        if input_dims:
            break
    modalities: List[str] = list(input_dims.keys())
    if not modalities:
        raise RuntimeError("Ни в одном примере не найдено ни одной модальности")

    # ── 1. Строим модель и оптимизатор ─────────────────────────────────
    model = SupraMultitaskModel(
        input_dims=input_dims,
        hidden_dim=cfg.hidden_dim,
        emo_out_dim=7,
        pkl_out_dim=5,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                   weight_decay=1e-5)

    # ── 2. Сквозные эпохи (alignment → guidance → concept) ────────────
    for epoch in range(cfg.num_epochs):
        print(f"\n═══ Epoch {epoch + 1}/{cfg.num_epochs} ═══")

        # ─ Alignment stage ────────────────────────────────────────────
        model.train()
        for mod in modalities:
            total_loss, seen = 0.0, 0
            for batch in tqdm(mm_loader, desc=f"[Align] {mod}"):
                feat = first_non_none_feature(batch, mod)
                if feat is None:
                    continue

                mini = {
                    "features": {mod: feat.to(device)},
                    "labels":   move_labels_to_device(batch["labels"], device),
                    "modality": mod,
                }
                loss = alignment_train_step(
                    model, optimizer, mini,
                    beta1=cfg.beta1, beta2=cfg.beta2
                )
                total_loss += loss
                seen += 1
            avg = total_loss / max(seen, 1)
            print(f"    {mod}: align_loss = {avg:.4f}")

        # ─ Guidance set ───────────────────────────────────────────────
        loaders_dict = {m: mm_loader for m in modalities}  # всех кормим одним
        guidance = build_guidance_set(model, loaders_dict,
                                      top_k=cfg.top_k, device=device)
        print("    guidance_set built ✔")

        # ─ Concept-guided stage ───────────────────────────────────────
        for mod in modalities:
            total_loss, seen = 0.0, 0
            for batch in tqdm(mm_loader, desc=f"[Concept] {mod}"):
                feat = first_non_none_feature(batch, mod)
                if feat is None:
                    continue

                mini = {
                    "features": {mod: feat.to(device)},
                    "labels":   move_labels_to_device(batch["labels"], device),
                    "modality": mod,
                }
                loss = concept_guided_train_step(
                    model, optimizer, mini,
                    guidance_set=guidance,
                    lambda_=cfg.lambda_w,
                )
                total_loss += loss
                seen += 1
            avg = total_loss / max(seen, 1)
            print(f"    {mod}: concept_loss = {avg:.4f}")

    # ── 3. Сохраняем итоговый чекпойнт ────────────────────────────────
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    ckpt_path = Path(cfg.checkpoint_dir) / "supra_multitask_final.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"\n✔ Model saved to {ckpt_path.resolve()}")
