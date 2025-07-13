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

    # повторяем метки столько раз, сколько модальностей
    y_emo_rep = torch.cat([y_emo] * len(feats_by_mod), dim=0)   # [M·B, 7]
    valid_mask = ~torch.isnan(y_emo_rep).all(dim=1)
    if not valid_mask.any():
        return 0.0
    y_emo_rep = torch.nan_to_num(y_emo_rep)

    # ───── losses ─────
    optimizer.zero_grad()

    loss_task  = F.binary_cross_entropy_with_logits(
        logits_e[valid_mask], y_emo_rep[valid_mask])

    loss_ortho = orthogonality_loss(z_emo, z_aux)

    class_ids  = y_emo_rep.argmax(dim=1)
    loss_contr = cross_modal_triplet_loss(
        z_emo[valid_mask],
        class_ids[valid_mask],
        modal_ids[valid_mask],
        margin,
    )

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
    def _calc_k(n: int) -> int:
        """Считаем целое k по правилу выше и заодно страхуемся от нулей."""
        k = int(round(n * top_k)) if isinstance(top_k, float) and top_k < 1 \
            else int(top_k)
        return max(1, min(k, n))   # минимум 1, не больше n

    model.eval()
    bank = {
        "emotion": {c: [] for c in range(7)},
        "personality": {}
    }

    for mod, loader in loaders_by_mod.items():
        # для PKL
        pkl_feat = {(i, b): [] for i in range(5) for b in [0, 1]}
        pkl_err  = {(i, b): [] for i in range(5) for b in [0, 1]}

        for batch in tqdm(loader, desc=f"[guidance] {mod}"):
            feats = {k: v.to(device) for k, v in batch["features"].items()}
            y_e   = batch["labels"]["emotion"].to(device)
            y_p   = batch["labels"]["personality"].to(device)

            out          = model(feats, mod)
            z_emo, z_aux = out["z_emo"], out["z_aux"]
            logits_e     = out["emotion_logits"]
            preds_p      = out["personality_scores"]

            # ---------- EMOTION ----------
            probs = torch.softmax(logits_e, dim=1)
            for cls in range(7):
                mask = (~torch.isnan(y_e[:, cls])) & (y_e[:, cls] > 0.5)
                if mask.any():
                    k = _calc_k(mask.sum().item())
                    top_idx = probs[mask, cls].topk(k).indices
                    bank["emotion"][cls].extend(z_emo[mask][top_idx].cpu())

            # ---------- PERSONALITY ----------
            mse = F.mse_loss(preds_p, y_p, reduction='none')    # [B,5]
            for t in range(5):
                col_y, col_mse = y_p[:, t], mse[:, t]
                valid = ~torch.isnan(col_y)
                for b, mask in [(0, valid & (col_y < 0.5)),
                                (1, valid & (col_y >= 0.5))]:
                    if mask.any():
                        pkl_feat[(t, b)].extend(z_aux[mask].cpu())
                        pkl_err[(t, b)].extend(col_mse[mask].cpu())

        # --- finalize PKL bank for current modality ---
        bank["personality"][mod] = {}
        for key, feats in pkl_feat.items():
            if feats:
                feats = torch.stack(feats)
                errs  = torch.tensor(pkl_err[key])
                k = _calc_k(errs.size(0))
                top = errs.topk(k, largest=False).indices
                bank["personality"][mod][key] = feats[top]
            else:
                bank["personality"][mod][key] = torch.empty(0, z_aux.size(1))

    # tensor-ify emotion lists
    bank["emotion"] = {
        c: (torch.stack(lst) if lst else torch.empty(0, z_emo.size(1)))
        for c, lst in bank["emotion"].items()
    }
    return bank


# ═════════════════════ CONCEPT-GUIDED STEP ═══════════════════════════
def concept_guided_train_step(model: nn.Module,
                              optimizer: torch.optim.Optimizer,
                              batch: dict,
                              guidance_set: dict,
                              lambda_: float = 0.5) -> float:
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

    total_loss = 0.0
    # ---------- Personality ----------
    if valid_p.any():
        total_loss += F.mse_loss(preds_p[valid_p], y_p[valid_p])
        for trait in range(5):
            pos_mask = y_p[valid_p][:, trait] >= 0.5
            for b, mask in [(1, pos_mask), (0, ~pos_mask)]:
                if mask.sum():
                    bank = guidance_set["personality"][modality].get((trait, b))
                    if bank is not None and bank.numel():
                        rand = torch.randint(0, bank.size(0), (mask.sum(),))
                        total_loss += lambda_ * similarity_loss(
                            z_aux[valid_p][mask], bank[rand].to(device))

    # ---------- Emotion ----------
    if valid_e.any():
        total_loss += F.binary_cross_entropy_with_logits(
            logits_e[valid_e], y_e[valid_e])
        y_bin = (y_e[valid_e] > 0.5).int()
        for cls in range(7):
            idx = (y_bin[:, cls] == 1).nonzero(as_tuple=True)[0]
            bank = guidance_set["emotion"][cls]
            if idx.numel() and bank.numel():
                rand = torch.randint(0, bank.size(0), (idx.size(0),))
                total_loss += lambda_ * similarity_loss(
                    z_emo[valid_e][idx], bank[rand].to(device))

    total_loss.backward()
    optimizer.step()
    return float(total_loss)
