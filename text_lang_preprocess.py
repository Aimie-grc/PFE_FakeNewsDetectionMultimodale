"""Détection de langue (langdetect) + traduction vers l’anglais (M2M100), comme dans detect_english.ipynb."""
from __future__ import annotations

import os

import re

from langdetect import DetectorFactory, LangDetectException, detect, detect_langs
from transformers.utils import logging as hf_logging

DetectorFactory.seed = 0
hf_logging.set_verbosity_error()

# Indices très simples si langdetect n’a pas assez de signal (textes courts, posts, etc.)
_RE_FR_CHARS = re.compile(r"[àâçéèêëîïôùûüÿœæ]", re.I)
_RE_FR_WORDS = re.compile(
    r"\b(le|la|les|un|une|des|est|été|pour|dans|sur|avec|que|qui|pas|plus|très|faux|vrai|selon|annonce)\b",
    re.I,
)
_RE_EN_WORDS = re.compile(
    r"\b(the|and|was|were|this|that|with|from|breaking|news|said|report|according|fake|real|hi|hello|hey|thanks|thank you)\b",
    re.I,
)

MODEL_NAME = "facebook/m2m100_418M"

_tok = None
_model = None
_device = None


def _translation_device():
    import torch

    if os.environ.get("TRANSLATE_DEVICE", "").lower() in ("cuda", "gpu"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


def _ensure_translator():
    global _tok, _model, _device
    if _model is not None:
        return
    import torch
    from transformers import M2M100Config, M2M100ForConditionalGeneration, M2M100Tokenizer

    token = os.getenv("HF_TOKEN")
    _device = _translation_device()
    _tok = M2M100Tokenizer.from_pretrained(MODEL_NAME, token=token)
    cfg = M2M100Config.from_pretrained(MODEL_NAME, token=token)
    cfg.tie_word_embeddings = False
    _model = M2M100ForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        config=cfg,
        token=token,
    ).to(_device)
    _model.eval()


def _best_from_detect_langs(s: str, min_prob: float = 0.18) -> str | None:
    try:
        cands = detect_langs(s)
    except LangDetectException:
        return None
    if not cands:
        return None
    best = cands[0]
    return best.lang if best.prob >= min_prob else None


def _heuristic_lang(t: str) -> str | None:
    """Dernier recours : indices orthographiques / lexicaux fr vs en."""
    if _RE_FR_CHARS.search(t):
        return "fr"
    fr_hits = len(_RE_FR_WORDS.findall(t))
    en_hits = len(_RE_EN_WORDS.findall(t))
    if fr_hits >= 2 and fr_hits >= en_hits:
        return "fr"
    if en_hits >= 2 and en_hits > fr_hits:
        return "en"
    if fr_hits >= 1 and en_hits == 0:
        return "fr"
    if en_hits >= 1 and fr_hits == 0:
        return "en"
    return None


def detect_language(text: str) -> str | None:
    t = (text or "").strip()
    if not t:
        return None
    t = " ".join(t.split())

    if len(t) <= 2:
        return _heuristic_lang(t)

    # Textes courts : indices lexicaux avant langdetect (souvent faux sur 1 phrase)
    if len(t) <= 48:
        h = _heuristic_lang(t)
        if h:
            return h

    # Plusieurs passes : texte brut, répétition (n‑grams), variante courte/longue
    variants = [t]
    if len(t) < 120:
        variants.append(f"{t}\n{t}")
        variants.append(f"{t} {t} {t}")

    for s in variants:
        try:
            return detect(s)
        except LangDetectException:
            pass
        lg = _best_from_detect_langs(s)
        if lg:
            return lg

    return _heuristic_lang(t)


def translate_to_english(text: str, max_new_tokens: int = 256) -> str:
    """Traduit vers l’anglais. Si déjà anglais, renvoie le texte tel quel."""
    import torch

    t = (text or "").strip()
    if not t:
        return ""

    lang = detect_language(t)
    if lang is None:
        raise ValueError("Langue non détectable.")
    if lang == "en":
        return t

    _ensure_translator()
    if lang not in _tok.lang_code_to_id:
        raise ValueError(f"Langue non supportée par m2m100: {lang}")

    _tok.src_lang = lang
    encoded = _tok(t, return_tensors="pt").to(_device)
    with torch.no_grad():
        generated = _model.generate(
            **encoded,
            forced_bos_token_id=_tok.get_lang_id("en"),
            max_new_tokens=max_new_tokens,
        )
    return _tok.batch_decode(generated, skip_special_tokens=True)[0]


def prepare_text_for_inference(text: str, max_new_tokens: int = 256) -> dict:
    """Texte prêt pour DistilBERT anglais : détection + traduction si besoin.

    Retourne ``text`` (chaîne à tokeniser), ``language``, ``translated``, et optionnellement
    ``translation_note`` si la traduction a été ignorée.
    """
    t = (text or "").strip()
    if not t:
        return {"text": "", "language": None, "translated": False}

    lang = detect_language(t)
    if lang is None:
        return {
            "text": t,
            "language": None,
            "translated": False,
            "translation_note": "langue_non_detectee",
        }
    if lang == "en":
        return {"text": t, "language": "en", "translated": False}

    try:
        _ensure_translator()
        if lang not in _tok.lang_code_to_id:
            return {
                "text": t,
                "language": lang,
                "translated": False,
                "translation_note": f"langue_non_supportee_m2m100:{lang}",
            }
        out = translate_to_english(t, max_new_tokens=max_new_tokens)
        return {"text": out or t, "language": lang, "translated": True}
    except ValueError as e:
        return {
            "text": t,
            "language": lang,
            "translated": False,
            "translation_note": str(e),
        }


def load_translator_at_startup() -> None:
    """Précharge M2M100 au démarrage du serveur (évite le délai au 1er texte non anglais)."""
    _ensure_translator()
