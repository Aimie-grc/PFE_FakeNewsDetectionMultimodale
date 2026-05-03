"""API multipart (option 2): envoi image comme vrai fichier.

Lancer:
  python serve_multipart.py
ou
  uvicorn serve_multipart:app --host 0.0.0.0 --port 8001

Options env:
  ENABLE_FACE_COHERENCE=0 — désactive Qwen2-VL (cohérence personnes image/texte, très lourd en VRAM).
  INCLUDE_RAW_VLM=1 — inclut le JSON brut du VLM dans face_coherence.
"""
import io
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import RedirectResponse
from PIL import Image

from text_lang_preprocess import load_translator_at_startup, prepare_text_for_inference

W = Path(__file__).parent / "beeest_clip_rag_bert_200.pt"
app = FastAPI(title="Détecteur de fake news multimodal")


def _compute_face_coherence(img: Image.Image, article_text: str) -> dict | None:
    """Cohérence personnes image (Qwen2-VL) vs noms cités dans le texte. Désactiver : ENABLE_FACE_COHERENCE=0."""
    import os

    if os.environ.get("ENABLE_FACE_COHERENCE", "1").strip().lower() in ("0", "false", "no", "off"):
        return None
    try:
        from Qwen_face import get_service

        r = get_service().predict(img, article_text)
        slim = {
            "score_percent": r.get("score_percent"),
            "score": r.get("score"),
            "personnes": r.get("personnes"),
            "face_score": r.get("face_score"),
            "parse_error": r.get("parse_error"),
            "details": r.get("details"),
        }
        if os.environ.get("INCLUDE_RAW_VLM", "").strip().lower() in ("1", "true", "yes", "on"):
            slim["raw_vlm"] = r.get("raw_vlm")
        return slim
    except Exception as e:
        return {"error": str(e)}


@app.get("/")
def root():
    return RedirectResponse("/docs", status_code=307)


@app.on_event("startup")
def _load():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import models, transforms
    from transformers import DistilBertConfig, DistilBertModel, DistilBertTokenizer

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.text_encoder = DistilBertModel(DistilBertConfig())
            r = models.resnet50(weights=None)
            r.fc = nn.Identity()
            self.image_encoder = r
            self.text_proj = nn.Sequential(nn.Linear(768, 256))
            self.image_proj = nn.Sequential(nn.Linear(2048, 256))
            self.classifier = nn.Sequential(nn.Linear(513, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, 2))

        def forward(self, ids, mask, px):
            t = self.text_proj(self.text_encoder(ids, attention_mask=mask).last_hidden_state[:, 0])
            i = self.image_proj(self.image_encoder(px))
            c = (F.normalize(t, 1) * F.normalize(i, 1)).sum(1, True)
            return self.classifier(torch.cat([t, i, c], 1))

    d = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = Net().to(d)
    m.load_state_dict(torch.load(W, map_location=d, weights_only=False), strict=True)
    m.eval()
    app.state.inf = {
        "m": m,
        "tok": DistilBertTokenizer.from_pretrained("distilbert-base-uncased"),
        "tfm": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "D": d,
    }


@app.on_event("startup")
def _load_translator():
    """Précharge M2M100 (évite le gel au 1er texte non anglais). Désactiver : SKIP_M2M100_WARMUP=1."""
    import os

    if os.environ.get("SKIP_M2M100_WARMUP", "").strip().lower() in ("1", "true", "yes", "on"):
        return
    try:
        load_translator_at_startup()
    except Exception as e:
        print(f"[serve_multipart] Préchargement M2M100 ignoré: {e}")


@app.post("/predict")
async def predict(text: str = Form(...), image: UploadFile = File(...)):
    import torch

    prep = prepare_text_for_inference(text)
    text_model = prep["text"]

    i = app.state.inf
    raw = await image.read()
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    p = i["tfm"](img)[None].to(i["D"])
    e = i["tok"](text_model, return_tensors="pt", truncation=True, max_length=512, padding="max_length")
    ids = e["input_ids"].to(i["D"])
    mask = e["attention_mask"].to(i["D"])
    with torch.no_grad():
        o = i["m"](ids, mask, p)
    out = {
        "logits": o[0].tolist(),
        "class": int(o.argmax(-1).item()),
        "language": prep.get("language"),
        "text_translated": bool(prep.get("translated", False)),
    }
    if prep.get("translation_note"):
        out["translation_note"] = prep["translation_note"]

    try:
        from explicabilite import explain_single_sample

        exp = explain_single_sample(
            i["m"],
            {"input_ids": ids, "attention_mask": mask, "image": p},
            0,
        )
        out["model_explain"] = {
            "reason_auto": exp["reason_auto"],
            "contributions": exp["contributions"],
        }
    except Exception as ex:
        out["model_explain"] = {"error": str(ex)}

    # Texte original : meilleure extraction des noms (français, etc.) ; image déjà chargée
    fc = _compute_face_coherence(img, (text or "").strip())
    if fc is not None:
        out["face_coherence"] = fc

    return out
