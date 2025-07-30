import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# ======================================================================
#                             LOSSES
# ======================================================================
def orthogonality_loss(f_emo, f_aux):
    f_emo = F.normalize(f_emo, dim=-1)
    f_aux = F.normalize(f_aux, dim=-1)
    prod  = torch.matmul(f_emo.T, f_aux)
    return torch.norm(prod, p="fro") ** 2


def similarity_loss(f_uni, f_guided):
    return F.mse_loss(f_uni, f_guided)


def supervised_contrastive_loss(features_emo, labels_emo, margin=0.2):
    """
    features_emo : [N, D]
    labels_emo   : [N, C] (multi-hot, без NaN)
    """
    cos_sim    = F.cosine_similarity(features_emo[:, None, :],
                                     features_emo[None, :, :], dim=-1)
    lbl_ids    = labels_emo.argmax(dim=1)          # [N]
    pos_mask   = lbl_ids[:, None] == lbl_ids[None] # [N, N]
    neg_mask   = ~pos_mask

    loss = 0.0
    for i in range(cos_sim.size(0)):
        pos_sim = cos_sim[i][pos_mask[i]]
        neg_sim = cos_sim[i][neg_mask[i]]
        pos_loss = torch.logsumexp(-pos_sim, dim=0) if pos_sim.numel() else 0.0
        neg_loss = torch.logsumexp( neg_sim, dim=0) if neg_sim.numel() else 0.0
        loss += pos_loss + neg_loss

    return loss / cos_sim.size(0)


# ======================================================================
#                    ALIGNMENT  (эмоции + контраст)
# ======================================================================
def alignment_train_step(
        model,
        optimizer,
        batch,
        beta1: float = 1.0,
        beta2: float = 1.0
):
    """
    batch:
        features      – dict модальностей
        modality      – str ('image', 'text', ...)
        labels.emotion      – [B, 7]  (NaN, если метки нет)
    """
    x_dict   = batch["features"]
    modality = batch["modality"]
    y_e      = batch["labels"]["emotion"].to(next(model.parameters()).device)

    # -- маска «валидных» эмо-меток (где хоть одна не-NaN)
    valid_rows = ~torch.isnan(y_e).all(dim=1)
    if not valid_rows.any():
        # в этой мини-партии нет эмо-меток → градиент не нужен
        return 0.0

    # заменяем NaN на 0, чтобы BCE не ругался
    y_e      = torch.nan_to_num(y_e)

    optimizer.zero_grad()
    out      = model(x_dict, modality)
    f_emo    = out["z_emo"]
    f_aux    = out["z_aux"]
    logits   = out["emotion_logits"]

    # --- task loss (BCE только по валидным строкам)
    loss_task = F.binary_cross_entropy_with_logits(
        logits[valid_rows], y_e[valid_rows]
    )

    # --- доп. лоссы
    loss_orth  = orthogonality_loss(f_emo, f_aux)
    loss_align = supervised_contrastive_loss(
        f_emo[valid_rows], y_e[valid_rows]
    )

    loss = loss_task + beta1 * loss_orth + beta2 * loss_align
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def build_guidance_set(model, dataloaders_by_modality, top_k=20, device="cuda"):
    model.eval()
    guidance_set = {"emotion": {}, "personality": {}}

    for modality, loader in dataloaders_by_modality.items():
        emo_guides = {i: [] for i in range(7)}  # 7 эмоций
        pkl_guides = {(i, b): [] for i in range(5) for b in [0, 1]}  # 5 признаков * 2 (low/high)

        emo_scores = {i: [] for i in range(7)}
        pkl_errors = {(i, b): [] for i in range(5) for b in [0, 1]}

        for batch in tqdm(loader):
            x_dict = {k: v.to(device) for k, v in batch["features"].items()}
            y_e = batch["labels"]["emotion"].to(device)  # [B, 7]
            y_p = batch["labels"]["personality"].to(device)  # [B, 5]

            out = model(x_dict, modality)
            z_emo = out["z_emo"]           # [B, D]
            z_aux = out["z_aux"]           # [B, D]
            logits = out["emotion_logits"] # [B, 7]
            preds = out["personality_scores"]  # [B, 5]

            # === EMOTION ===
            probs = torch.softmax(logits, dim=1)  # [B, 7]
            for i in range(7):
                y_e_i = y_e[:, i]
                valid_mask = (~torch.isnan(y_e_i)) & (y_e_i > 0.5)
                if valid_mask.any():
                    emo_scores[i].extend(probs[valid_mask, i].cpu())
                    emo_guides[i].extend(z_emo[valid_mask].cpu())

            # === PERSONALITY ===
            err = F.mse_loss(preds, y_p, reduction='none')  # [B, 5]
            for i in range(5):
                y_p_i = y_p[:, i]
                err_i = err[:, i]
                valid_mask = ~torch.isnan(y_p_i)

                low_mask = valid_mask & (y_p_i < 0.5)
                high_mask = valid_mask & (y_p_i >= 0.5)

                for mask, bin_val in [(low_mask, 0), (high_mask, 1)]:
                    if mask.any():
                        z_batch = z_aux[mask]
                        err_batch = err_i[mask]
                        pkl_guides[(i, bin_val)].extend(z_batch.cpu())
                        pkl_errors[(i, bin_val)].extend(err_batch.cpu())

        # === Select top-K emotion ===
        guidance_set["emotion"][modality] = {}
        for i in range(7):
            if emo_guides[i]:
                guides = torch.stack(emo_guides[i])            # [N, D]
                scores = torch.stack(emo_scores[i])            # [N]
                k = min(int(top_k), scores.size(0))
                top = scores.topk(k).indices                   # [K]
                guidance_set["emotion"][modality][i] = guides[top]
            else:
                guidance_set["emotion"][modality][i] = torch.empty(0, z_emo.size(1))

        # === Select top-K personality ===
        guidance_set["personality"][modality] = {}
        for key, feats in pkl_guides.items():
            if feats:
                feats = torch.stack(feats)                     # [N, D]
                errs = torch.tensor(pkl_errors[key])           # [N]
                k = min(int(top_k), errs.size(0))
                top = errs.topk(k, largest=False).indices      # наименьшая ошибка
                guidance_set["personality"][modality][key] = feats[top]
            else:
                guidance_set["personality"][modality][key] = torch.empty(0, z_aux.size(1))

    return guidance_set


def concept_guided_train_step(model, optimizer, batch, guidance_set, lambda_):
    x_dict = batch["features"]
    modality = batch["modality"]
    labels = batch["labels"]
    device = next(model.parameters()).device

    y_p = labels["personality"].to(device)
    y_e = labels["emotion"].to(device)

    valid_p = ~torch.isnan(y_p).all(dim=1)
    valid_e = ~torch.isnan(y_e).all(dim=1)

    optimizer.zero_grad()
    out = model(x_dict, modality)
    f_aux = out["z_aux"]
    f_emo = out["z_emo"]
    preds_p = out["personality_scores"]
    logits_e = out["emotion_logits"]

    loss = 0.0

    # --- Personality loss + guidance
    if valid_p.any():
        loss_task_p = F.mse_loss(preds_p[valid_p], y_p[valid_p])
        loss += loss_task_p

        for i in range(5):
            bin_mask = y_p[valid_p][:, i] >= 0.5
            for bin_val in [0, 1]:
                mask = (bin_mask if bin_val else ~bin_mask)
                if mask.sum() == 0:
                    continue
                bank = guidance_set["personality"][modality].get((i, bin_val))
                if bank is not None and bank.numel() > 0:
                    rand_idx = torch.randint(0, bank.size(0), (mask.sum(),))
                    guided = bank[rand_idx].to(device)
                    sim_loss = similarity_loss(f_aux[valid_p][mask], guided)
                    loss += lambda_ * sim_loss

    # --- Emotion loss + guidance
    if valid_e.any():
        # y_e_label = y_e[valid_e].argmax(dim=1)
        loss_task_e = F.binary_cross_entropy_with_logits(logits_e[valid_e], y_e[valid_e])
        loss += loss_task_e

        y_e_bin = (y_e[valid_e] > 0.5).int()
        for i in range(7):
            idxs = (y_e_bin[:, i] == 1).nonzero(as_tuple=True)[0]
            if idxs.numel() == 0:
                continue
            bank = guidance_set["emotion"][modality].get(i)
            if bank is not None and bank.numel() > 0:
                rand_idx = torch.randint(0, bank.size(0), (idxs.size(0),))
                guided = bank[rand_idx].to(device)
                sim_loss = similarity_loss(f_emo[valid_e][idxs], guided)
                loss += lambda_ * sim_loss

    loss.backward()
    optimizer.step()
    return loss.item()
