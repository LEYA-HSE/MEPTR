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

import os, logging
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

    if improved_emo or improved_pkl:
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
    cfg          – объект-конфиг (поля: device, random_seed, num_epochs,
                    hidden_dim, lr, beta1, beta2, lambda_w, top_k,
                    checkpoint_dir)
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

    # ───────────────── 2-A. ALIGNMENT-фаза ─────────────────────
    for epoch in range(cfg.align_epochs):
        logging.info(f"═══ ALIGNMENT {epoch + 1}/{cfg.align_epochs} ═══")
        model.train()

        total_loss, steps = 0.0, 0
        for batch in tqdm(mm_loader, desc="[Align]"):
            loss = alignment_train_step(
                model, optimizer, batch,
                beta_ortho=cfg.beta_ortho,
                beta_contr=cfg.beta_contr,
                margin=cfg.triplet_margin,
            )
            total_loss += loss;  steps += 1
        logging.info(f"[Align] mean-loss {total_loss / steps:.4f}")

        # — Dev / Test (как раньше) —
        if dev_loaders:
            _log_dev_and_stop (
                model,
                dev_loaders,
                test_loaders,
                device,
                epoch,
                cfg.align_epochs + cfg.replay_epochs,
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
                cfg=cfg)
    # ───────────────── 2-B. GUIDANCE-банк ──────────────────────
    guidance = build_guidance_set(
        model,
        loaders_by_mod={m: mm_loader for m in modalities},
        top_k=cfg.top_k,
        device=device,
    )
    logging.info("Guidance bank collected ✅")

    # Замораживаем shared-encoder
    for p in model.shared_encoder.parameters():
        p.requires_grad_(False)
    logging.info("Shared encoder frozen ✅")

    # ───────────────── 2-C. REPLAY / CONCEPT-guided ────────────
    for epoch in range(cfg.replay_epochs):
        logging.info(f"═══ REPLAY {epoch + 1}/{cfg.replay_epochs} ═══")
        model.train()

        for mod in modalities:
            tl, steps = 0.0, 0
            for batch in tqdm(mm_loader, desc=f"[Concept] {mod}"):
                feat = first_non_none_feature(batch, mod)
                if feat is None:
                    continue
                mini = {
                    "features": {mod: feat.to(device)},
                    "labels": move_labels_to_device(batch["labels"], device),
                    "modality": mod,
                }
                loss = concept_guided_train_step(
                    model, optimizer, mini,
                    guidance_set=guidance,
                    lambda_=cfg.lambda_w,
                    gamma_task=cfg.gamma_task
                )
                tl += loss; steps += 1
            logging.info(f"[Concept] {mod}: mean-loss {tl / max(steps, 1):.4f}")

        # — Dev / Test + early-stopping (тот же код) —
        if dev_loaders:
            stop = _log_dev_and_stop(
                model, dev_loaders, test_loaders, device,
                cfg.align_epochs + epoch,
                cfg.align_epochs + cfg.replay_epochs,
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


    # ── 2. Сквозные эпохи (alignment → guidance → concept) ────────────
    # for epoch in range(cfg.num_epochs):
    #     logging.info(f"═══ Epoch {epoch + 1}/{cfg.num_epochs} ═══")
    #     model.train()

    #     # ─ Alignment stage ────────────────────────────────────────────
    #     total_loss, seen = 0.0, 0
    #     for batch in tqdm(mm_loader, desc="[Align] mix"):
    #         # batch уже содержит ВСЕ модальности из collate_fn
    #         loss = alignment_train_step(
    #             model,
    #             optimizer,
    #             batch,                               # передаём как есть
    #             beta_ortho = cfg.beta_ortho,         # новые имена параметров
    #             beta_contr = cfg.beta_contr,
    #             margin     = cfg.triplet_margin
    #         )
    #         total_loss += loss
    #         seen += 1
    #     avg = total_loss / max(seen, 1)
    #     logging.info(f"[Align] mixed: loss={avg:.4f}")

    #     # ─ Guidance set ───────────────────────────────────────────────
    #     loaders_dict = {m: mm_loader for m in modalities}  # всех кормим одним
    #     guidance = build_guidance_set(model, loaders_dict,
    #                                   top_k=cfg.top_k, device=device)
    #     logging.info("guidance_set built ✔")

    #     # ─ FREEZE SHARED ENCODER ──────────────────────────────────────
    #     # Делаем это один раз – после Alignment-фазы, перед первой Concept-guided
    #     if epoch == 0:
    #         for param in model.shared_encoder.parameters():      # <-- имя слоя
    #             param.requires_grad_(False)
    #         logging.info("Shared encoder parameters are frozen ✔")

    #     # ─ Concept-guided stage ───────────────────────────────────────
    #     for mod in modalities:
    #         total_loss, seen = 0.0, 0
    #         for batch in tqdm(mm_loader, desc=f"[Concept] {mod}"):
    #             feat = first_non_none_feature(batch, mod)
    #             if feat is None:
    #                 continue

    #             mini = {
    #                 "features": {mod: feat.to(device)},
    #                 "labels":   move_labels_to_device(batch["labels"], device),
    #                 "modality": mod,
    #             }
    #             loss = concept_guided_train_step(
    #                 model, optimizer, mini,
    #                 guidance_set=guidance,
    #                 lambda_=cfg.lambda_w,
    #             )
    #             total_loss += loss
    #             seen += 1
    #         avg = total_loss / max(seen, 1)
    #         logging.info(f"[Concept] {mod}: loss={avg:.4f}")

    #     # ----- Validation -----------------------------------------
    #     if dev_loaders is not None:
    #         dev_emotion_avgs, dev_pkl_avgs = [], []
    #         logging.info(f"\n—— Dev metrics ——")


    #         for ds_name, dev_loader in dev_loaders.items():
    #             m = evaluate_epoch(model, dev_loader, device)

    #             # ----- EMOTION -----
    #             if {"mF1", "mUAR"} <= m.keys():
    #                 emo_avg = (m["mF1"] + m["mUAR"]) / 2
    #                 dev_emotion_avgs.append(emo_avg)
    #             else:
    #                 emo_avg = None         # в этом датасете эмоций нет

    #             # ----- PERSONALITY --
    #             if {"ACC", "CCC"} <= m.keys():
    #                 pkl_avg = (m["ACC"] + m["CCC"]) / 2
    #                 dev_pkl_avgs.append(pkl_avg)
    #             else:
    #                 pkl_avg = None         # в этом датасете PKL нет

    #             # лог
    #             msg = " · ".join(f"{k}:{v:.4f}" for k, v in m.items())
    #             logging.info(
    #                 f"[Dev:{ds_name}] {msg}"
    #                 + (f" · emo_avg:{emo_avg:.4f}" if emo_avg is not None else "")
    #                 + (f" · pkl_avg:{pkl_avg:.4f}" if pkl_avg is not None else "")
    #             )

    #         # --- средние только по тем, что были ---
    #         epoch_dev_emo = float(np.mean(dev_emotion_avgs)) if dev_emotion_avgs else None
    #         epoch_dev_pkl = float(np.mean(dev_pkl_avgs))     if dev_pkl_avgs else None
    #         logging.info(
    #             "Mean Dev | "
    #             + (f"emo_avg={epoch_dev_emo:.4f} " if epoch_dev_emo is not None else "")
    #             + (f"pkl_avg={epoch_dev_pkl:.4f}"  if epoch_dev_pkl  is not None else "")
    #         )

    #         # --- early-stopping логика ---
    #         improved_emo = True if epoch_dev_emo is None else epoch_dev_emo > best_emotion_avg
    #         improved_pkl = True if epoch_dev_pkl is None else epoch_dev_pkl > best_pkl_avg

    #         # Обнуляем общий счётчик, если улучшилась хотя бы одна ветка
    #         if improved_emo or improved_pkl:
    #             if epoch_dev_emo is not None: best_emotion_avg = epoch_dev_emo
    #             if epoch_dev_pkl is not None: best_pkl_avg     = epoch_dev_pkl
    #             patience_counter = 0

    #             # сохраняем чекпойнт
    #             os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    #             ckpt_path = Path(cfg.checkpoint_dir) / (
    #                 f"best_ep{epoch+1}"
    #                 + (f"_emo{epoch_dev_emo:.4f}" if epoch_dev_emo is not None else "")
    #                 + (f"_pkl{epoch_dev_pkl:.4f}" if epoch_dev_pkl is not None else "")
    #                 + ".pt"
    #             )
    #             torch.save(model.state_dict(), ckpt_path)
    #             logging.info(f"✔ Best model saved: {ckpt_path.name}")

    #         else:
    #             patience_counter += 1
    #             reasons = []
    #             if not improved_emo: reasons.append("emotion")
    #             if not improved_pkl: reasons.append("pkl")
    #             logging.warning(
    #                 f"No improvement in {', '.join(reasons)} — "
    #                 f"patience {patience_counter}/{cfg.max_patience}"
    #             )
    #             if patience_counter >= cfg.max_patience:
    #                 logging.info(f"Early stopping at epoch {epoch + 1}/{cfg.num_epochs}")
    #                 break

    #     # ─── Test after each epoch ─────────────────────────────────────
    #     if test_loaders is not None:
    #         logging.info("\n—— Test metrics ——")
    #         for ds_name, test_loader in test_loaders.items():
    #             m = evaluate_epoch(model, test_loader, device)
    #             msg = " · ".join(f"{k}:{v:.4f}" for k, v in m.items())
    #             logging.info(f"  🏁 Test[{ds_name}] | {msg}")
