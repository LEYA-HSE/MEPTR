# coding: utf-8
"""
supra_train_utils.py  – все лоссы, шаги обучения и строительство guidance-банка
для «чистой» supra-modal архитектуры (eq.(3)+(7) из статьи).
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# ═════════════════════════════ LOSSES ════════════════════════════════
def orthogonality_loss(emo_vec: torch.Tensor,
                       aux_vec: torch.Tensor) -> torch.Tensor:
    """
    Frobenius-норма произведения Fᵀ_emo · F_aux  – штраф за перекрытие
    эмо-пространства с не-эмо-пространством.

    Args:
        emo_vec: [N, D]  – эмоциональные векторы supra-энкодера.
        aux_vec: [N, D]  – векторы non-emotion энкодера.

    Returns:
        Скаляр-тензор – значение ортогонального штрафа.
    """
    emo_norm = F.normalize(emo_vec, dim=-1)
    aux_norm = F.normalize(aux_vec, dim=-1)
    return torch.norm(emo_norm.T @ aux_norm, p="fro") ** 2


def similarity_loss(pred_vec: torch.Tensor, guide_vec: torch.Tensor) -> torch.Tensor:
    """
    L₂-приближение унимодального вектора к guidance-якорю.

    Args:
        pred_vec:  [B, D] – предсказанные фичи.
        guide_vec: [B, D] – случайно выбранные «якоря» того же класса.

    Returns:
        Скаляр-тензор mse-ошибки.
    """
    return F.mse_loss(pred_vec, guide_vec)

def binarize_with_nan(x, threshold=0.5):
    # Создаем маску NaN
    nan_mask = torch.isnan(x)

    # Бинаризуем (не затрагивая NaN)
    binary = torch.zeros_like(x)
    binary[x >= threshold] = 1.0

    # Восстанавливаем NaN там, где они были
    binary[nan_mask] = float('nan')

    return binary


def cross_modal_triplet_loss(emo_vec: torch.Tensor,
                             class_ids: torch.Tensor,
                             modal_ids: torch.Tensor,
                             margin: float = 0.2) -> torch.Tensor:
    """
    Triplet-контраст (eq.(3) из статьи):

        L = max(0, margin − cos(i,j) + cos(i,k))
        • (i, j) – одна эмоция, **разные** модальности;
        • (i, k) – одна модальность, **разные** эмоции.

    Args:
        emo_vec:   [N, D] – нормированные supra-векторы.
        class_ids: [N]    – id эмоции (0…6).
        modal_ids: [N]    – id модальности  (0=image, 1=text, …).
        margin:    гиперпараметр.

    Returns:
        Среднее по всем найденным триплетам.
    """
    emo_norm = F.normalize(emo_vec, dim=-1)
    cosine   = emo_norm @ emo_norm.T                            # [N, N]

    loss, triplets = 0.0, 0
    for i in range(emo_vec.size(0)):
        pos_mask = (class_ids == class_ids[i]) & (modal_ids != modal_ids[i])
        neg_mask = (class_ids != class_ids[i]) & (modal_ids == modal_ids[i])
        if pos_mask.any() and neg_mask.any():
            loss += F.relu(
                margin
                - cosine[i, pos_mask].mean()
                + cosine[i, neg_mask].mean()
            )
            triplets += 1
    return loss / max(triplets, 1)


# ═══════════════════════════ ALIGNMENT STEP ══════════════════════════
def alignment_train_step(model: nn.Module,
                         optimizer: torch.optim.Optimizer,
                         batch: dict,
                         beta_ortho: float = 0.05,
                         beta_contr: float = 0.10,
                         margin: float = 0.2) -> float:
    """
    Один оптимизационный шаг «Alignment»-фазы.

    batch может быть:
      • «монобатч» – ровно одна модальность; тогда modal_ids = 0;
      • «микси-батч» – несколько модальностей (для triplet-лосса).

    Args:
        model:      supra-модель.
        optimizer:  любой оптимизатор.
        batch: {
            'features': {mod: Tensor[B,Dₘ], ...},
            'labels'  : {'emotion': Tensor[N,7]},
            'mod_ids' : Optional[Tensor[N]]  – если DataLoader уже склеил.
        }
        beta_ortho: вес ортогонального штрафа.
        beta_contr: вес triplet-контраста.
        margin:     margin в triplet.

    Returns:
        Скаляр – значение суммарного loss за этот шаг (для логов).
    """
    device = next(model.parameters()).device
    feats_by_mod = {m: v.to(device) for m, v in batch["features"].items()}
    y_emo        = batch["labels"]["emotion"].to(device)        # [B,7]

    # ───── forward по всем модальностям отдельно ─────
    z_emo_lst, z_aux_lst, logits_lst, modal_ids = [], [], [], []
    for idx, (mod_key, tensor) in enumerate(feats_by_mod.items()):
        out = model({mod_key: tensor}, modality=mod_key)        # один проход
        z_emo_lst .append(out["z_emo"])                         # [B, H]
        z_aux_lst .append(out["z_aux"])                         # [B, H]
        logits_lst.append(out["emotion_logits"])                # [B, 7]
        modal_ids .append(torch.full((tensor.size(0),), idx,    # [B]
                                     dtype=torch.long, device=device))

    # ───── собираем назад (уже одинаковые размеры) ─────
    z_emo    = torch.cat(z_emo_lst,  dim=0)                     # [M·B, H]
    z_aux    = torch.cat(z_aux_lst,  dim=0)                     # [M·B, H]
    logits_e = torch.cat(logits_lst, dim=0)                     # [M·B, 7]
    modal_ids = torch.cat(modal_ids, dim=0)                     # [M·B]

    # print(y_emo)

    # повторяем метки столько раз, сколько модальностей
    y_emo_rep = torch.cat([y_emo] * len(feats_by_mod), dim=0)   # [M·B, 7]
    y_emo_rep = binarize_with_nan(y_emo_rep, threshold=0.01)
    valid_mask = ~torch.isnan(y_emo_rep).all(dim=1)
    if not valid_mask.any():
        return 0.0
    y_emo_rep = torch.nan_to_num(y_emo_rep)

    # ───── losses ─────
    optimizer.zero_grad()

    loss_task  = F.binary_cross_entropy_with_logits(
        logits_e[valid_mask], y_emo_rep[valid_mask], weight=torch.FloatTensor([ 5.890161, 7.534918,  11.228363,  27.722221,   1.3049748,  5.6189237, 26.639517 ]).to(device))
    # loss_task  = F.cross_entropy(
    #     logits_e[valid_mask], y_emo_rep[valid_mask], weight=torch.FloatTensor([ 5.890161, 7.534918,  11.228363,  27.722221,   1.3049748,  5.6189237, 26.639517 ]).to(device))

    loss_ortho = orthogonality_loss(z_emo, z_aux)

    # вариант 1
    class_ids  = y_emo_rep.argmax(dim=1)
    loss_contr = cross_modal_triplet_loss(
            z_emo[valid_mask],
            class_ids[valid_mask],
            modal_ids[valid_mask],
            margin,
        )

    # # вариант 2
    # loss_contr = 0.0
    # for class_id in range(7):
    #     loss_contr += cross_modal_triplet_loss(
    #         z_emo[valid_mask],
    #         y_emo_rep[:, class_id][valid_mask],
    #         modal_ids[valid_mask],
    #         margin,
    #     )

    total_loss = loss_task + beta_ortho * loss_ortho + beta_contr * loss_contr
    total_loss.backward()
    optimizer.step()
    return float(total_loss)

# ═════════════════════ BUILD GUIDANCE SET (v4) ═══════════════════════
@torch.no_grad()
def build_guidance_set(model: nn.Module,
                       loaders_by_mod: dict[str, torch.utils.data.DataLoader],
                       top_k: int = 8,
                       device: str = "cuda") -> dict:
    """
    Сбор «ядерного» guidance-банка:
      • Emotion – глобальный:  {class_id: [≤K, D]}
      • Personality – тот же per-modality банк, что был у тебя.

    Returns:
        {
          'emotion'    : {0: Tensor[n₀,D], …, 6: Tensor[n₆,D]},
          'personality': {mod: {(trait,b): Tensor[n,D], …}, …}
        }
    """
    # ─── helper ──────────────────────────────────────────────────────
    def _k_from_ratio(n: int) -> int:
        if isinstance(top_k, float) and top_k < 1:
            k = int(round(n * top_k))
        else:
            k = int(top_k)
        return max(1, min(k, n))

    model.eval()
    bank = {
        "emotion":     {c: [] for c in range(7)},
        "personality": {(t, b): [] for t in range(5) for b in (0, 1)},
    }


    for mod, loader in loaders_by_mod.items():
        for batch in tqdm(loader, desc=f"[guidance] {mod}"):
            feats = {m: x.to(device) for m, x in batch["features"].items()}
            y_e   = batch["labels"]["emotion"].to(device)
            y_p   = batch["labels"]["personality"].to(device)

            out = model(feats, modality=mod)
            z_emo, z_aux = out["z_emo"], out["z_aux"]
            logits_e     = out["emotion_logits"]
            preds_p      = out["personality_scores"]

            # ---------- EMOTION ----------
            probs = torch.softmax(logits_e, dim=1)
            for cls in range(7):
                mask = (~torch.isnan(y_e[:, cls])) & (y_e[:, cls] > 0)
                if mask.any():
                    k = _k_from_ratio(mask.sum().item())
                    top = probs[mask, cls].topk(k).indices
                    bank["emotion"][cls].extend(z_emo[mask][top].cpu())

            # ---------- PERSONALITY ----------
            mse = F.mse_loss(preds_p, y_p, reduction='none')          # [B,5]
            for t in range(5):
                col_y, col_mse = y_p[:, t], mse[:, t]
                valid = ~torch.isnan(col_y)
                for b, m in ((0, valid & (col_y < 0.5)),
                             (1, valid & (col_y >= 0.5))):
                    if m.any():
                        k = _k_from_ratio(m.sum().item())
                        top = col_mse[m].topk(k, largest=False).indices
                        bank["personality"][(t, b)].extend(z_aux[m][top].cpu())

        # --- finalize PKL bank for current modality ---
        for cls in range(7):
            vecs = bank["emotion"][cls]
            bank["emotion"][cls] = torch.stack(vecs) if vecs else torch.empty(0, z_emo.size(1))

        for key, vecs in bank["personality"].items():
            bank["personality"][key] = torch.stack(vecs) if vecs else torch.empty(0, z_aux.size(1))

        return bank


# ═════════════════════ CONCEPT-GUIDED STEP ═══════════════════════════
def concept_guided_train_step(
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        batch: dict,
        guidance_set: dict,
        lambda_: float = 0.5,
        gamma_task: float = 0.1
    ) -> float:
    """
    Fine-tune унимодальной ветки + similarity к guidance-якорям.
    Shared-encoder перед вызовом должен быть **заморожен**.

    Args:
        batch – «обычный» (одна модальность).
        lambda_ – вес similarity-штрафа.

    Returns:
        Суммарный loss (float).
    """
    device   = next(model.parameters()).device
    x_dict   = batch["features"]
    modality = batch["modality"]
    y_p      = batch["labels"]["personality"].to(device)
    y_e      = batch["labels"]["emotion"].to(device)

    valid_p = ~torch.isnan(y_p).all(dim=1)
    valid_e = ~torch.isnan(y_e).all(dim=1)

    optimizer.zero_grad()
    out   = model(x_dict, modality)
    z_aux, z_emo = out["z_aux"], out["z_emo"]
    preds_p, logits_e = out["personality_scores"], out["emotion_logits"]

    task_loss, sim_loss = 0.0, 0.0
    # ---------- Personality ----------
    if valid_p.any():
        task_loss += F.mse_loss(preds_p[valid_p], y_p[valid_p])     # ◄─ task
        for trait in range(5):
            pos_mask = y_p[valid_p][:, trait] >= 0.5
            for b, mask in ((1, pos_mask), (0, ~pos_mask)):
                if mask.sum():
                    anchors = guidance_set["personality"][(trait, b)]
                    if anchors.numel():
                        k = min(8, mask.sum().item(), anchors.size(0))
                        rand = torch.randint(0, anchors.size(0), (k,))
                        sim_loss += similarity_loss(                 # ◄─ sim
                            z_aux[valid_p][mask][:k], anchors[rand].to(device))

    # ---------- Emotion ----------
    if valid_e.any():
        task_loss += F.binary_cross_entropy_with_logits(            # ◄─ task
            logits_e[valid_e], binarize_with_nan(y_e, threshold=0.01)[valid_e], weight=torch.FloatTensor([ 5.890161, 7.534918,  11.228363,  27.722221,   1.3049748,  5.6189237, 26.639517 ]).to(device))
        # task_loss += F.cross_entropy(
        # logits_e[logits_e], y_e[valid_e], weight=torch.FloatTensor([ 5.890161, 7.534918,  11.228363,  27.722221,   1.3049748,  5.6189237, 26.639517 ]).to(device))
        y_bin = (y_e[valid_e] > 0).int()
        for cls in range(7):
            idx = (y_bin[:, cls] == 1).nonzero(as_tuple=True)[0]
            if idx.numel():
                anchors = guidance_set["emotion"][cls]
                if anchors.numel():
                    k = min(8, idx.numel(), anchors.size(0))
                    rand = torch.randint(0, anchors.size(0), (k,))
                    sim_loss += similarity_loss(                     # ◄─ sim
                        z_emo[valid_e][idx][:k], anchors[rand].to(device))

    # ----- финальный лосс c весами γ и λ -----
    total_loss = gamma_task * task_loss + lambda_ * sim_loss
    # --------------------------------------------------------------------
    total_loss.backward()
    optimizer.step()
    return float(total_loss)
