
# coding: utf-8
from __future__ import annotations

import os, logging
import random
from pathlib import Path
from typing import Dict, List
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from lion_pytorch import Lion

from utils.schedulers import SmartScheduler
from utils.logger_setup import color_metric, color_split
from utils.measures import mf1, uar, acc_func, ccc
from utils.losses import MultiTaskLossWithNaN
from models.models import MultiModalFusionModel, MultiModalFusionModelWithAblation


# ─────────────────────────────── utils ────────────────────────────────
def seed_everything(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
                   device: torch.device) -> Dict[str, float]:
    """Собирает метрики на всём лоадере."""
    model.eval()
    emo_preds, emo_tgts = [], []
    pkl_preds, pkl_tgts = [], []

    for batch in tqdm(loader, desc="Eval", leave=False):
        out = model(batch)

        # Emotion
        logits_e = out["emotion_logits"]
        y_e = batch["labels"]["emotion"]
        valid_e = ~torch.isnan(y_e).all(dim=1)
        if valid_e.any():
            p, t = process_predictions(logits_e[valid_e], y_e[valid_e])
            emo_preds.extend(p)
            emo_tgts.extend(t)

        # Personality
        preds_p = out["personality_scores"].cpu()
        y_p = batch["labels"]["personality"]
        valid_p = ~torch.isnan(y_p).all(dim=1)
        if valid_p.any():
            pkl_preds.append(preds_p[valid_p].numpy())
            pkl_tgts.append(y_p[valid_p].numpy())

    metrics: dict[str, float] = {}
    if emo_tgts:
        tgt, prd = np.asarray(emo_tgts), np.asarray(emo_preds)
        metrics["mF1"] = mf1(tgt, prd)
        metrics["mUAR"] = uar(tgt, prd)
    if pkl_tgts:
        tgt, prd = np.vstack(pkl_tgts), np.vstack(pkl_preds)
        metrics["ACC"] = acc_func(tgt, prd)
        metrics["CCC"] = ccc(tgt, prd)
    return metrics


def log_and_aggregate_split(name: str,
                            loaders: dict[str, DataLoader],
                            model: torch.nn.Module,
                            device: torch.device) -> dict[str, float]:
    """
    Универсальная функция логирования и подсчёта агрегатов для dev/test.
    """
    logging.info(f"—— {name} metrics ——")
    all_metrics: dict[str, float] = {}

    for ds_name, loader in loaders.items():
        m = evaluate_epoch(model, loader, device)
        all_metrics.update({f"{k}_{ds_name}": v for k, v in m.items()})
        # msg = " · ".join(f"{k}:{v:.4f}" for k, v in m.items())
        msg = " · ".join(color_metric(k, v) for k, v in m.items())
        logging.info(f"[{color_split(name)}:{ds_name}] {msg}")

    mf1s = [v for k, v in all_metrics.items() if k.startswith("mF1_")]
    uars = [v for k, v in all_metrics.items() if k.startswith("mUAR_")]
    accs = [v for k, v in all_metrics.items() if k.startswith("ACC_")]
    cccs = [v for k, v in all_metrics.items() if k.startswith("CCC_")]

    if mf1s and uars:
        all_metrics["mean_emo"] = float(np.mean(mf1s + uars))
    if accs and cccs:
        all_metrics["mean_pkl"] = float(np.mean(accs + cccs))

    if "mean_emo" in all_metrics or "mean_pkl" in all_metrics:
        summary_parts = []
        if "mean_emo" in all_metrics:
            summary_parts.append(color_metric("mean_emo", all_metrics["mean_emo"]))
        if "mean_pkl" in all_metrics:
            summary_parts.append(color_metric("mean_pkl", all_metrics["mean_pkl"]))
        logging.info(f"{name} Summary | " + " ".join(summary_parts))

    return all_metrics

# ────────────────────────── основной train() ──────────────────────────
def train(cfg,
          mm_loader: DataLoader,
          dev_loaders: dict[str, DataLoader] | None = None,
          test_loaders: dict[str, DataLoader] | None = None):
    """
    cfg          – объект-конфиг
    mm_loader    – train-DataLoader (multimodal)
    dev_loader   – optional, для валидации
    test_loader  – optional, для оценки в конце
    """

    seed_everything(cfg.random_seed)
    device = cfg.device

    # ── 0. Модель и оптимизатор ───────────────────────────────────────
    model = MultiModalFusionModelWithAblation(
        hidden_dim=cfg.hidden_dim,
        num_heads = cfg.num_transformer_heads,
        dropout=cfg.dropout,
        emo_out_dim=7,
        pkl_out_dim=5,
        device=device,
        ablation_config={"disabled_modalities": [], "disable_guide_emo": False, "disable_guide_pkl": True}
    ).to(device)

    if cfg.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == "lion":
        optimizer = Lion(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum)
    elif cfg.optimizer == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=cfg.lr)
    else:
        raise ValueError(f"⛔ Неизвестный оптимизатор: {cfg.optimizer}")

    logging.info(f"⚙️ Оптимизатор: {cfg.optimizer}, learning rate: {cfg.lr}")

    # --- Scheduler ---
    steps_per_epoch = sum(1 for b in mm_loader if b is not None)
    scheduler = SmartScheduler(
        scheduler_type=cfg.scheduler_type,
        optimizer=optimizer,
        config=cfg,
        steps_per_epoch=steps_per_epoch
    )

    # --- Loss ---
    criterion = MultiTaskLossWithNaN(
        weight_emotion=cfg.weight_emotion,
        weight_personality=cfg.weight_pers,
        emo_weights=(torch.FloatTensor(
            [5.890161, 7.534918, 11.228363, 27.722221,
             1.3049748, 5.6189237, 26.639517]).to(device)
                     if cfg.flag_emo_weight else None),
        personality_loss_type=cfg.pers_loss_type,
        emotion_loss_type=cfg.emotion_loss_type
    )

    best_dev, best_test = {}, {}
    best_mean_combo  = -float("inf")
    patience_counter = 0

    # ── 1. Эпохи ──────────────────────────────────────────────────────
    for epoch in range(cfg.num_epochs):
        logging.info(f"═══ EPOCH {epoch + 1}/{cfg.num_epochs} ═══")
        model.train()

        total_loss = 0.0
        total_samples = 0
        total_preds_emo, total_targets_emo = [], []
        total_preds_per, total_targets_per = [], []

        for batch in tqdm(mm_loader):
            if batch is None:
                continue

            emo_labels = batch["labels"]["emotion"].to(device)
            per_labels = batch["labels"]["personality"].to(device)

            valid_emo = ~torch.isnan(emo_labels).all(dim=1)
            valid_per = ~torch.isnan(per_labels).all(dim=1)

            outputs = model(batch)
            loss = criterion(outputs, {
                "emotion": emo_labels,
                "personality": per_labels,
                "valid_emo": valid_emo,
                "valid_per": valid_per
            })
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            scheduler.step(batch_level=True)

            bs = emo_labels.shape[0]
            total_loss += loss.item() * bs
            total_samples += bs

            preds_emo, targets_emo = process_predictions(
                outputs['emotion_logits'][valid_emo],
                emo_labels[valid_emo]
            )
            total_preds_emo.extend(preds_emo)
            total_targets_emo.extend(targets_emo)

            preds_per = outputs['personality_scores'][valid_per]
            targets_per = per_labels[valid_per]
            total_preds_per.extend(preds_per.cpu().detach().numpy().tolist())
            total_targets_per.extend(targets_per.cpu().detach().numpy().tolist())

        # --- train метрики ---
        train_loss = total_loss / total_samples
        mF1_train = mf1(total_targets_emo, total_preds_emo)
        mUAR_train = uar(total_targets_emo, total_preds_emo)
        logging.info(
            f"[{color_split('TRAIN')}] Loss={train_loss:.4f}, "
            f"UAR={mUAR_train:.4f}, MF1={mF1_train:.4f}, "
            f"MEAN={np.mean([mUAR_train, mF1_train]):.4f}"
        )

        # ── Evaluation ──
        cur_dev = log_and_aggregate_split("Dev", dev_loaders, model, device)
        cur_test = log_and_aggregate_split("Test", test_loaders, model, device) if test_loaders else {}

        cur_eval = cur_dev if cfg.early_stop_on == "dev" else cur_test

        mean_emo = cur_eval.get("mean_emo")
        mean_pkl = cur_eval.get("mean_pkl", 0.0)


        if mean_emo is not None and mean_pkl is not None:
            mean_combo = 0.5 * (mean_emo + mean_pkl)
        else:
            mean_combo = mean_emo if mean_emo is not None else mean_pkl  # фоллбэк на одну из метрик

        logging.info(f"[{color_split('COMBO')}] mean_emo={mean_emo:.4f}, mean_pkl={mean_pkl:.4f}, combo={mean_combo:.4f}")

        scheduler.step(mean_combo)

        # improved_emo = (mean_emo is not None) and (mean_emo > best_mean_emo)
        improved_combo = mean_combo > best_mean_combo

        if improved_combo:
            best_mean_combo  = mean_combo
            best_dev = cur_dev
            best_test = cur_test
            patience_counter = 0

            os.makedirs(cfg.checkpoint_dir, exist_ok=True)
            ckpt_path = Path(cfg.checkpoint_dir) / (
                f"best_ep{epoch + 1}_emo{mean_emo:.4f}_pkl{mean_pkl:.4f}.pt"
            )
            torch.save(model.state_dict(), ckpt_path)
            logging.info(f"✔ Best model saved: {ckpt_path.name}")
        else:
            patience_counter += 1
            logging.warning(f"No improvement — patience {patience_counter}/{cfg.max_patience}")
            if patience_counter >= cfg.max_patience:
                logging.info(f"Early stopping at epoch {epoch + 1}")
                break

    return best_dev, best_test
