import torch
import torch.nn as nn
import torch.nn.functional as F


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


# ======================================================================
#    СТРОИМ guidance-сет (отдельно эмоции и личностные факторы)
# ======================================================================
@torch.no_grad()
def build_guidance_set(model, dataloaders_by_modality, top_k=0.2, device="cuda"):
    """
    Возвращает:
        {
          'emotion':     {modality: [N, D_emo]},
          'personality': {modality: [M, D_aux]},
        }
    """
    model.eval()
    guidance_set = {"emotion": {}, "personality": {}}

    for modality, loader in dataloaders_by_modality.items():
        feats_e  = []   # эмо-фичи
        conf_e   = []   # confidence (эмо)
        lbls_e   = []

        feats_p  = []   # aux-фичи (личн.)
        errs_p   = []   # MSE error (личн.)

        for batch in loader:
            x_dict = {k: v.to(device) for k, v in batch["features"].items()}
            y_e    = batch["labels"]["emotion"].to(device)
            y_p    = batch["labels"]["personality"].to(device)

            out    = model(x_dict, modality)
            f_emo = out["z_emo"]          # [B, D_emo]
            f_aux = out["z_aux"]          # [B, D_aux]
            logits = out["emotion_logits"]
            preds  = out["personality_scores"]

            # ---------- EMOTION ----------
            valid_e = ~torch.isnan(y_e).all(dim=1)
            if valid_e.any():
                probs = torch.sigmoid(logits[valid_e])
                true_cls = y_e[valid_e].argmax(dim=1, keepdim=True)
                conf_e.append(torch.gather(probs, 1, true_cls).squeeze())
                feats_e.append(f_emo[valid_e])
                lbls_e.append(y_e[valid_e])

            # ---------- PERSONALITY -------
            valid_p = ~torch.isnan(y_p).all(dim=1)
            if valid_p.any():
                feat_batch = f_aux[valid_p]
                err_batch  = F.mse_loss(preds[valid_p], y_p[valid_p],
                                        reduction="none").mean(dim=1)
                feats_p.append(feat_batch)
                errs_p.append(err_batch)

        # ----- собрали всё -----
        if feats_e:
            feats_e  = torch.cat(feats_e)
            lbls_e   = torch.cat(lbls_e)
            conf_e   = torch.cat(conf_e)

            selected = []
            for c in range(lbls_e.size(1)):
                idx = (lbls_e[:, c] == 1).nonzero(as_tuple=True)[0]
                if idx.numel() == 0:
                    continue
                k = max(1, int(top_k * len(idx)))
                top_idx = idx[conf_e[idx].topk(k).indices]
                selected.append(feats_e[top_idx])
            guidance_set["emotion"][modality] = (
                torch.cat(selected) if selected else torch.empty(0, feats_e.size(1))
            )
        else:
            guidance_set["emotion"][modality] = torch.empty(0, 1)

        if feats_p:
            feats_p = torch.cat(feats_p)
            errs_p  = torch.cat(errs_p)
            k = max(1, int(top_k * len(errs_p)))
            top_idx = errs_p.topk(k, largest=False).indices
            guidance_set["personality"][modality] = feats_p[top_idx]
        else:
            guidance_set["personality"][modality] = torch.empty(0, 1)

    return guidance_set


# ======================================================================
#             CONCEPT-GUIDED (личностные факторы)
# ======================================================================
def concept_guided_train_step(model, optimizer, batch, guidance_set, lambda_):
    """
    batch:
        labels.personality – [B, 5]  (NaN-вектор если меток нет)
    """
    x_dict   = batch["features"]
    modality = batch["modality"]
    y_p      = batch["labels"]["personality"].to(next(model.parameters()).device)

    valid_p = ~torch.isnan(y_p).all(dim=1)
    if not valid_p.any():
        # нет валидных PKL-меток → скипаем (но всё-таки прогоняем вперёд,
        # чтобы BN / dropout вели себя корректно)
        with torch.no_grad():
            model(x_dict, modality)
        return 0.0

    optimizer.zero_grad()
    out    = model(x_dict, modality)
    f_aux  = out["z_aux"]                 # [B, D_aux]
    preds  = out["personality_scores"]    # [B, 5]

    # --- основной task-лосс
    loss_task = F.mse_loss(preds[valid_p], y_p[valid_p])

    # --- концептуальный гайд
    if guidance_set and guidance_set["personality"][modality].numel() > 0:
        g_bank = guidance_set["personality"][modality].to(f_aux.device)
        rand_idx   = torch.randint(0, g_bank.size(0), (f_aux.size(0),))
        f_guided   = g_bank[rand_idx]
    else:
        f_guided = f_aux.detach()

    loss_sim = similarity_loss(f_aux, f_guided)
    loss = loss_task + lambda_ * loss_sim
    loss.backward()
    optimizer.step()
    return loss.item()
