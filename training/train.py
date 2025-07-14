
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
    from train import train
"""
from __future__ import annotations

import os, logging
import random
from pathlib import Path
from typing import Dict, List
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.measures import mf1, uar, acc_func, ccc
from utils.losses import MultiTaskLossWithNaN
from models.models import MultiModalFusionModel


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
        out   = model(batch)

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

def _log_dev_and_stop(model, dev_loaders, test_loaders, device,
                      cur_epoch, total_epochs,
                      best_vals, patience,
                      counter_ref, patience_counter_ref, cfg):
    """
    Один вьюх-метод, чтобы не дублировать длинный код в обоих фазах.
    best_vals          – (best_emotion_avg, best_pkl_avg)
    counter_ref        – [getter_best_emo, getter_best_pkl, setter_best_emo, setter_best_pkl]
    patience_counter_ref – [getter_pc, setter_pc]
    """
    best_emo_get, best_pkl_get, best_emo_set, best_pkl_set = counter_ref
    pc_get, pc_set = patience_counter_ref

    dev_emo_avgs, dev_pkl_avgs = [], []
    logging.info("\n—— Dev metrics ——")
    for ds_name, dev_loader in dev_loaders.items():
        m = evaluate_epoch(model, dev_loader, device)

        # emotion
        emo_avg = ((m["mF1"] + m["mUAR"]) / 2) if {"mF1", "mUAR"} <= m.keys() else None
        if emo_avg is not None:
            dev_emo_avgs.append(emo_avg)

        # personality
        pkl_avg = ((m["ACC"] + m["CCC"]) / 2) if {"ACC", "CCC"} <= m.keys() else None
        if pkl_avg is not None:
            dev_pkl_avgs.append(pkl_avg)

        msg = " · ".join(f"{k}:{v:.4f}" for k, v in m.items())
        logging.info(f"[Dev:{ds_name}] {msg}"
                     + (f" · emo_avg:{emo_avg:.4f}" if emo_avg is not None else "")
                     + (f" · pkl_avg:{pkl_avg:.4f}" if pkl_avg is not None else ""))

    mean_emo = float(np.mean(dev_emo_avgs)) if dev_emo_avgs else None
    mean_pkl = float(np.mean(dev_pkl_avgs)) if dev_pkl_avgs else None
    logging.info("Mean Dev | "
                 + (f"emo_avg={mean_emo:.4f} " if mean_emo is not None else "")
                 + (f"pkl_avg={mean_pkl:.4f}" if mean_pkl is not None else ""))

    improved_emo = True if mean_emo is None else mean_emo > best_emo_get()
    improved_pkl = True if mean_pkl is None else mean_pkl > best_pkl_get()

    if improved_emo and improved_pkl:
        if mean_emo is not None:
            best_emo_set(mean_emo)
        if mean_pkl is not None:
            best_pkl_set(mean_pkl)
        pc_set(0)

        os.makedirs(cfg.checkpoint_dir, exist_ok=True)
        ckpt_path = Path(cfg.checkpoint_dir) / (
            f"best_ep{cur_epoch+1}"
            + (f"_emo{mean_emo:.4f}" if mean_emo is not None else "")
            + (f"_pkl{mean_pkl:.4f}" if mean_pkl is not None else "")
            + ".pt"
        )
        torch.save(model.state_dict(), ckpt_path)
        logging.info(f"✔ Best model saved: {ckpt_path.name}")
    else:
        pc_set(pc_get() + 1)
        reasons = []
        if not improved_emo: reasons.append("emotion")
        if not improved_pkl: reasons.append("pkl")
        logging.warning(f"No improvement in {', '.join(reasons)} — "
                        f"patience {pc_get()}/{patience}")
        if pc_get() >= patience:
            logging.info(f"Early stopping at epoch {cur_epoch + 1}/{total_epochs}")
            return True  # stop training

    # — Test —
    if test_loaders:
        logging.info("\n—— Test metrics ——")
        for ds_name, test_loader in test_loaders.items():
            m = evaluate_epoch(model, test_loader, device)
            msg = " · ".join(f"{k}:{v:.4f}" for k, v in m.items())
            logging.info(f"  🏁 Test[{ds_name}] | {msg}")
    return False

# ────────────────────────── основной train() ──────────────────────────
def train(cfg,
          mm_loader:   DataLoader,
          dev_loaders:  dict[str, DataLoader] | None = None,
          test_loaders: dict[str, DataLoader] | None = None):
    """
    cfg          – объект-конфиг
    mm_loader    – train-DataLoader (multimodal)
    dev_loader   – optional, для валидации
    test_loader  – optional, для оценки в конце
    """

    seed_everything(cfg.random_seed)
    device = cfg.device

    # prepare early-stopping
    best_emotion_avg = -float("inf")
    best_pkl_avg     = -float("inf")
    patience_counter = 0

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


    # ── 1. Строим модель и оптимизатор ─────────────────────────────────
    model = MultiModalFusionModel(
        hidden_dim=cfg.hidden_dim,
        emo_out_dim=7,
        pkl_out_dim=5,
        device = device
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                   weight_decay=cfg.weight_decay)

    criterion = MultiTaskLossWithNaN(weight_emotion=cfg.weight_emotion,
                                weight_personality=cfg.weight_pers,
                                emo_weights=torch.FloatTensor([ 5.890161,7.534918,11.228363,27.722221,1.3049748,5.6189237,26.639517]).to(device),
                                personality_loss_type=cfg.pers_loss_type)

    for epoch in range(cfg.num_epochs):
        logging.info(f"═══ EPOCHS {epoch + 1}/{cfg.num_epochs} ═══")

        model.train()

        total_loss = 0.0
        total_samples = 0
        total_preds_emo = []
        total_targets_emo = []
        total_preds_per = []
        total_targets_per = []

        for batch in tqdm(mm_loader):
            if batch is None:
                continue

            emo_labels = batch["labels"]["emotion"].to(device)
            per_labels = batch["labels"]["personality"].to(device)

            valid_emo = ~torch.isnan(emo_labels).all(dim=1)
            valid_per = ~torch.isnan(per_labels).all(dim=1)

            outputs = model(batch)

            loss = criterion(outputs, {"emotion": emo_labels, "personality": per_labels, 'valid_emo': valid_emo, 'valid_per': valid_per})
            # print(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            bs = emo_labels.shape[0]
            total_loss += loss.item() * bs
            total_samples += bs

            preds_emo, terget_emo =  process_predictions(outputs['emotion_logits'][valid_emo], emo_labels[valid_emo])
            total_preds_emo.extend(preds_emo)
            total_targets_emo.extend(terget_emo)
            preds_per = outputs['personality_scores'][valid_per]
            terget_per = per_labels[valid_per]
            total_preds_per.extend(preds_per.cpu().detach().numpy().tolist())
            total_targets_per.extend(terget_per.cpu().detach().numpy().tolist())

        train_loss = total_loss / total_samples
        uar_m = uar(total_targets_emo, total_preds_emo)
        mf1_m = mf1(total_targets_emo, total_preds_emo)

        mean_train = np.mean([uar_m, mf1_m])

        logging.info(
            f"[TRAIN] Loss={train_loss:.4f}, UAR={uar_m:.4f}, MF1={mf1_m:.4f}, "
            f"MEAN={mean_train:.4f}")

        # — Dev / Test + early-stopping —
        if dev_loaders:
            stop = _log_dev_and_stop(
                model, dev_loaders, test_loaders, device,
                epoch, cfg.num_epochs,
                best_vals=(best_emotion_avg, best_pkl_avg),
                patience=cfg.max_patience,
                counter_ref=[
                    lambda: best_emotion_avg,
                    lambda: best_pkl_avg,
                    lambda v: globals().update(best_emotion_avg=v),
                    lambda v: globals().update(best_pkl_avg=v)
                    ],
                patience_counter_ref=[
                    lambda: patience_counter,
                    lambda v: globals().update(patience_counter=v)
                    ],
                cfg=cfg
            )

            if stop:
                break
