# ---------- Explicabilité (contributions + raison automatique) ----------
# Aligné sur le forward de serve_multipart / serve.py :
#   c = (F.normalize(t, 1) * F.normalize(i, 1)).sum(1, True)
#   logits = classifier(cat(t, i, c))
# Pas de modèle CLIP dans ce graphe : le 3e signal est la fusion texte–image (scalaire c).

from __future__ import annotations

import torch
import torch.nn.functional as F

# Seuil empirique sur le scalaire de fusion c : en dessous, on parle d’alignement faible
# (c dépend de la normalisation L1 ; ajuster si besoin après observation sur tes batches).
FUSION_LOW_THRESHOLD = 0.05


def _normalize_to_percent(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    total = sum(max(v, 0.0) for v in values.values())
    if total <= 0:
        n = len(values)
        return {k: round(100.0 / n, 1) for k in values} if n else {}

    out = {k: 100.0 * max(v, 0.0) / total for k, v in values.items()}
    rounded = {k: round(v, 1) for k, v in out.items()}
    diff = round(100.0 - sum(rounded.values()), 1)
    first_key = next(iter(rounded))
    rounded[first_key] = round(rounded[first_key] + diff, 1)
    return rounded


def explain_single_sample(
    model: torch.nn.Module,
    batch: dict,
    idx: int,
    *,
    fusion_low_threshold: float = FUSION_LOW_THRESHOLD,
) -> dict:
    """Interprétation locale pour un élément ``idx`` du batch.

    ``batch`` doit contenir au moins : ``input_ids``, ``attention_mask``, ``image`` (tenseurs).

    Retourne des contributions agrégées (BERT / Image / Fusion) et une phrase d’explication.
    Les scores de contribution sont des parts % positives dérivées des normes L2 de ``t``, ``i``
    et de |c| (scalaire de fusion), à titre **heuristique** (pas des vrais attributions intégrées).
    """
    model.eval()
    with torch.no_grad():
        text_outputs = model.text_encoder(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        text_feature = model.text_proj(text_outputs.last_hidden_state[:, 0, :])
        image_feature = model.image_proj(model.image_encoder(batch["image"]))

        t_n = F.normalize(text_feature, p=1, dim=-1)
        i_n = F.normalize(image_feature, p=1, dim=-1)
        fusion = (t_n * i_n).sum(dim=1, keepdim=True)

        fused = torch.cat([text_feature, image_feature, fusion], dim=1)
        logits = model.classifier(fused)
        probs = torch.softmax(logits, dim=1)

    bert_score = torch.norm(text_feature[idx], p=2).item()
    image_score = torch.norm(image_feature[idx], p=2).item()
    fusion_scalar = float(fusion[idx].item())
    fusion_strength = abs(fusion_scalar)

    contributions = _normalize_to_percent(
        {
            "BERT": bert_score,
            "Fusion": fusion_strength,
            "Image": image_score,
        }
    )

    dominant = max(contributions, key=contributions.get)

    if dominant == "BERT":
        reason = "Le texte (représentation BERT) domine l’explication heuristique de la décision."
    elif dominant == "Fusion" and abs(fusion_scalar) < fusion_low_threshold:
        reason = (
            "Le terme de fusion texte–image est faible : faible alignement des représentations "
            "multimodales (heuristique)."
        )
    elif dominant == "Fusion":
        reason = (
            "Le signal de fusion texte–image (produit des embeddings normalisés L1) pèse fort "
            "dans cette explication heuristique."
        )
    else:
        reason = "Les indices issus de la branche image (ResNet) dominent cette explication heuristique."

    pred_class = int(torch.argmax(probs[idx]).item())
    pred_fake_prob = float(probs[idx, 1].item())

    return {
        "prediction_class": pred_class,
        "prob_fake": pred_fake_prob,
        "fusion_scalar": fusion_scalar,
        "contributions": contributions,
        "reason_auto": reason,
        # Alias rétrocompat pour d’anciens scripts qui lisaient clip_similarity
        "clip_similarity": fusion_scalar,
    }


# Compatibilité : ancien nom de constante
CLIP_LOW_SIM_THRESHOLD = FUSION_LOW_THRESHOLD
