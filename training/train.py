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
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.measures import mf1, uar, acc_func, ccc
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

def transform_matrix(matrix):
    threshold1 = 1 - 1/7
    threshold2 = 1/7
    mask1 = matrix[:, 0] >= threshold1
    result = np.zeros_like(matrix[:, 1:])
    transformed = (matrix[:, 1:] >= threshold2).astype(int)
    result[~mask1] = transformed[~mask1]
    return result

def process_predictions(pred_emo, true_emo):
    pred_emo = torch.nn.functional.softmax(pred_emo, dim=1).cpu().detach().numpy()
    pred_emo = transform_matrix(pred_emo).tolist()
    true_emo = true_emo.cpu().detach().numpy()
    true_emo = np.where(true_emo > 0, 1, 0)[:, 1:].tolist()
    return pred_emo, true_emo


# ─────────────────────────── evaluation ────────────────────────────
@torch.no_grad()
def evaluate_epoch(model: torch.nn.Module,
                   loader: DataLoader,
                   device   : torch.device) -> Dict[str, float]:
    """
    Прогоняет loader, собирает метрики.
    Возврат: dict {metric: value}
    """
    model.eval()

    emo_preds, emo_tgts = [], []
    pkl_preds, pkl_tgts = [], []

    for batch in loader:
        feats = {m: (v.to(device) if v is not None else None)
                 for m, v in batch["features"].items()}
        out   = model(feats, modality=list(feats.keys())[0])

        # raw logits → process_predictions → бинарные вектора без NaN
        logits_e = out["emotion_logits"]          # [B,7]  (на GPU)
        y_e      = batch["labels"]["emotion"]     # [B,7]  (на CPU/Nan)

        valid_e = ~torch.isnan(y_e).all(dim=1)
        if valid_e.any():
            y_pred_proc, y_true_proc = process_predictions(
                logits_e[valid_e],          # raw logits
                y_e     [valid_e]           # ground-truth one-hot (NaN-free)
            )
            emo_preds.extend(y_pred_proc)
            emo_tgts .extend(y_true_proc)

        # personality  (как раньше)
        preds_p = out["personality_scores"].cpu()
        y_p     = batch["labels"]["personality"]
        valid_p = ~torch.isnan(y_p).all(dim=1)
        if valid_p.any():
            pkl_preds.append(preds_p[valid_p].numpy())
            pkl_tgts .append(y_p   [valid_p].numpy())

    # --- агрегация ---
    metrics = {}
    if emo_tgts:
        tgt = np.asarray(emo_tgts); prd = np.asarray(emo_preds)
        metrics["mF1"]  = mf1(tgt, prd)
        metrics["mUAR"] = uar(tgt, prd)
    if pkl_tgts:
        tgt = np.vstack(pkl_tgts); prd = np.vstack(pkl_preds)
        metrics["ACC"] = acc_func(tgt, prd)
        metrics["CCC"] = ccc(tgt, prd)

    return metrics


# ────────────────────────── основной train() ──────────────────────────
def train(cfg,
          mm_loader:   DataLoader,
          dev_loaders:  dict[str, DataLoader] | None = None,
          test_loaders: dict[str, DataLoader] | None = None):
    """
    cfg          – объект-конфиг (поля: device, random_seed, num_epochs,
                    hidden_dim, lr, beta1, beta2, lambda_w, top_k,
                    checkpoint_dir)
    mm_loader    – train-DataLoader (multimodal)
    dev_loader   – optional, для валидации
    test_loader  – optional, для оценки в конце
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
    if not input_dims:
        raise RuntimeError("Ни в одном примере не найдено ни одной модальности")

    modalities: List[str] = list(input_dims.keys())

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
        model.train()

        # ─ Alignment stage ────────────────────────────────────────────
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

            # ----- Validation -----------------------------------------
            if dev_loaders is not None:
                print("\n—— Dev metrics ——")
                for ds_name, dev_loader in dev_loaders.items():
                    m = evaluate_epoch(model, dev_loader, device)
                    msg = " · ".join(f"{k}:{v:.4f}" for k, v in m.items())
                    print(f"  🔎 Dev[{ds_name}] | {msg}")

    # ── 3. Сохраняем итоговый чекпойнт ────────────────────────────────
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    ckpt_path = Path(cfg.checkpoint_dir) / "supra_multitask_final.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"\n✔ Model saved to {ckpt_path.resolve()}")

    # ─ 4. Финальный тест ------------------------------------------
    if test_loaders is not None:
        print("\n—— Test metrics ——")
        for ds_name, test_loader in test_loaders.items():
            m = evaluate_epoch(model, test_loader, device)
            msg = " · ".join(f"{k}:{v:.4f}" for k, v in m.items())
            print(f"  🏁 Test[{ds_name}] | {msg}")
