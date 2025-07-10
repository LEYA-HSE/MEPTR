import torch
import torch.nn as nn
import torch.nn.functional as F


# === LOSS ===
def orthogonality_loss(f_emo, f_aux):
    f_emo = F.normalize(f_emo, dim=-1)
    f_aux = F.normalize(f_aux, dim=-1)
    prod = torch.matmul(f_emo.T, f_aux)
    return torch.norm(prod, p='fro') ** 2

def similarity_loss(f_uni, f_guided):
    return F.mse_loss(f_uni, f_guided)

def supervised_contrastive_loss(features_emo, labels_emo, margin=0.2):
    cos_sim = F.cosine_similarity(features_emo.unsqueeze(1), features_emo.unsqueeze(0), dim=-1)
    labels_emo = labels_emo.argmax(dim=1)
    pos_mask = labels_emo.unsqueeze(1) == labels_emo.unsqueeze(0)
    neg_mask = ~pos_mask

    loss = 0.0
    for i in range(cos_sim.size(0)):
        pos_sim = cos_sim[i][pos_mask[i]]
        neg_sim = cos_sim[i][neg_mask[i]]

        pos_loss = torch.logsumexp(-pos_sim, dim=0) if len(pos_sim) > 0 else 0.0
        neg_loss = torch.logsumexp(neg_sim, dim=0) if len(neg_sim) > 0 else 0.0
        loss += pos_loss + neg_loss

    return loss / cos_sim.size(0)


# === ЭТАП ВЫРАВНИВАНИЯ ===
def alignment_train_step(model, optimizer, batch):
    x_dict, y, modality = batch['features'], batch['labels'], batch['modality']
    optimizer.zero_grad()

    out = model(x_dict, modality)
    f_emo = out['z_emo']
    f_aux = out['z_aux']
    logits = out['emotion_logits']

    loss_task = F.binary_cross_entropy_with_logits(logits, y.float())
    loss_orth = orthogonality_loss(f_emo, f_aux)
    loss_align = supervised_contrastive_loss(f_emo, y)

    loss = loss_task + loss_orth + loss_align
    loss.backward()
    optimizer.step()
    return loss.item()


def build_guidance_set(model, dataloaders_by_modality, top_k=0.2, device="cuda"):
    model.eval()
    guidance_set = {'emotion': {}, 'personality': {}}

    with torch.no_grad():
        for modality, loader in dataloaders_by_modality.items():
            features_e, features_p, labels_e, errors_p, confidences_e = [], [], [], [], []
            for batch in loader:
                x_dict, y = batch['features'], batch['labels']
                x_dict = {k: v.to(device) for k, v in x_dict.items()}
                y = y.to(device)
                out = model(x_dict, modality)
                f_emo = out['z_emo']
                f_aux = out['z_aux']
                logits = out['emotion_logits']
                preds = out['personality_scores']

                # Emotion confidence
                probs = torch.sigmoid(logits)
                conf = torch.gather(probs, 1, y.argmax(dim=1, keepdim=True)).squeeze()
                confidences_e.append(conf)
                features_e.append(f_emo)
                labels_e.append(y)

                # PKL error
                error = F.mse_loss(preds, y, reduction='none').mean(dim=1)
                features_p.append(f_aux)
                errors_p.append(error)

            # Stack all
            feats_e = torch.cat(features_e)
            lbls_e = torch.cat(labels_e)
            confs_e = torch.cat(confidences_e)
            feats_p = torch.cat(features_p)
            errs_p = torch.cat(errors_p)

            # Top-k confident emotions
            selected_e = []
            for c in range(lbls_e.size(1)):
                idx = (lbls_e[:, c] == 1).nonzero(as_tuple=True)[0]
                if len(idx) == 0:
                    continue
                top_idx = idx[confs_e[idx].topk(max(1, int(top_k * len(idx))).indices)]
                selected_e.append(feats_e[top_idx])
            guidance_set['emotion'][modality] = torch.cat(selected_e) if selected_e else torch.empty(0, feats_e.size(1))

            # Top-k lowest error personalities
            top_idx = errs_p.topk(int(top_k * len(errs_p)), largest=False).indices
            guidance_set['personality'][modality] = feats_p[top_idx]

    return guidance_set


# === ОБУЧЕНИЕ С КОНЦЕПТУАЛЬНЫМ ГАЙДЕНСОМ ===
def concept_guided_train_step(model, optimizer, batch, guidance_set, lambda_):
    x_dict, y, modality = batch['features'], batch['labels'], batch['modality']
    optimizer.zero_grad()
    out = model(x_dict, modality)
    f_aux = out['z_aux']
    preds = out['personality_scores']

    loss_task = F.mse_loss(preds, y)

    if guidance_set and modality in guidance_set['personality'] and len(guidance_set['personality'][modality]) > 0:
        g = guidance_set['personality'][modality]
        idx = torch.randint(0, g.size(0), (f_aux.size(0),))
        f_guided = g[idx].to(f_aux.device)
    else:
        f_guided = f_aux.detach()

    loss_sim = similarity_loss(f_aux, f_guided)
    loss = loss_task + lambda_ * loss_sim
    loss.backward()
    optimizer.step()
    return loss.item()
